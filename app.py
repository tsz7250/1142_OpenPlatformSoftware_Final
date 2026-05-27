import base64
import csv
import hashlib
import hmac
import json
import logging
import math
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
import os
import re
import torch
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import requests
from flask import Flask, abort, jsonify, render_template, request
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

from sentence_transformers import util
try:
    from rapidfuzz import fuzz
except Exception:
    # Fallback using difflib when rapidfuzz isn't available.
    import difflib

    class _FuzzFallback:
        @staticmethod
        def WRatio(a: str, b: str) -> float:
            try:
                a = (a or "").lower()
                b = (b or "").lower()
                if not a or not b:
                    return 0.0
                return difflib.SequenceMatcher(None, a, b).ratio() * 100.0
            except Exception:
                return 0.0

        @staticmethod
        def token_set_ratio(a: str, b: str) -> float:
            try:
                a_tokens = sorted(set((a or "").lower().split()))
                b_tokens = sorted(set((b or "").lower().split()))
                a_join = " ".join(a_tokens)
                b_join = " ".join(b_tokens)
                return difflib.SequenceMatcher(None, a_join, b_join).ratio() * 100.0
            except Exception:
                return 0.0

        @staticmethod
        def partial_ratio(a: str, b: str) -> float:
            try:
                a = (a or "").lower()
                b = (b or "").lower()
                if len(a) == 0 or len(b) == 0:
                    return 0.0
                # approximate partial by comparing shorter to longer
                short, long = (a, b) if len(a) <= len(b) else (b, a)
                best = 0.0
                window = len(short)
                for i in range(0, len(long) - window + 1):
                    part = long[i : i + window]
                    best = max(best, difflib.SequenceMatcher(None, short, part).ratio())
                return best * 100.0
            except Exception:
                return 0.0

        @staticmethod
        def partial_token_sort_ratio(a: str, b: str) -> float:
            try:
                a_sorted = " ".join(sorted((a or "").lower().split()))
                b_sorted = " ".join(sorted((b or "").lower().split()))
                return _FuzzFallback.partial_ratio(a_sorted, b_sorted)
            except Exception:
                return 0.0

    fuzz = _FuzzFallback()

load_dotenv()

EMBEDDING_MODEL = "text-embedding-bge-m3"
VECTOR_SEARCHER: Optional["VectorSearcher"] = None
BM25_INDEX: Optional["BM25Index"] = None
DEBUG_CSV_PATH = os.getenv("DEBUG_CSV_PATH", "debug_candidates.csv").strip()

app = Flask(__name__)

import sys
# 設定/配置 root logger 以便將日誌正常輸出到 stdout
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.root.addHandler(stdout_handler)
logging.root.setLevel(logging.INFO)


@dataclass
class FaqRecord:
    record_id: str
    category: str
    question: str
    answer: str
    updated: str
    search_text: str
    synonyms: str = ""


class KeywordOutput(BaseModel):
    primary_keywords: List[str]
    secondary_keywords: List[str]


class ScoreEntry(BaseModel):
    id: str
    reasoning: str
    score: float


class RerankOutput(BaseModel):
    scores: List[ScoreEntry]


class CoverageOutput(BaseModel):
    reasoning_process: str
    is_multi_part_query: bool
    has_coverage: bool


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.strip().lower()
    return re.sub(r"\s+", " ", text)


def is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
    )


def tokenize_bm25(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    tokens = re.findall(r"[a-z0-9]+", text)
    # 找出所有連續的 CJK (中文) 片段，並在片段內部進行 Unigram 與 Bigram 斷詞
    cjk_pattern = r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df]+"
    cjk_chunks = re.findall(cjk_pattern, text)
    for chunk in cjk_chunks:
        # Unigram (單字)
        for char in chunk:
            tokens.append(char)
        # Bigram (雙字連詞)
        for i in range(len(chunk) - 1):
            tokens.append(chunk[i] + chunk[i + 1])
    return tokens


CONFLICT_GROUPS = [
    {"pph", "aep", "tw-supa"},
    {"申請人", "發明人", "專利代理人"},  # 關係人衝突：代理人、申請人與發明人為不同主體，提問與 FAQ 條目若角色不一致則判定為衝突
    {"發明", "新型", "設計"},            # 專利類型衝突：發明、新型與設計專利適用不同規定，若類型不一致則判定為衝突
    {"實體審查", "新型技術報告"},        # 審查程序衝突：實體審查與新型技術報告互斥，若不一致則判定為衝突
    {"更正", "補換發"},                  # 「專利權更正」vs「補換發專利證書」概念衝突
    {"延長", "延緩"},                    # 「專利權期間延長」vs「延緩實體審查」概念衝突
    {"年費", "申請規費"},                # 繳費類型衝突
]


def check_conflict(query_text: str, record_question: str) -> bool:
    q_lower = query_text.lower()
    r_lower = record_question.lower()
    
    for group in CONFLICT_GROUPS:
        # 1. 檢查用戶提問中是否包含此群組的成員
        q_present = {member for member in group if member in q_lower}
        
        # 如果用戶提問中包含同一衝突群組中的多個成員，代表用戶在詢問涵蓋多個角色的通用問題，不應觸發衝突
        if len(q_present) > 1:
            continue
            
        # 2. 如果用戶提問中包含此群組的成員，則繼續比對 FAQ 條目
        if q_present:
            # 3. 檢查 FAQ 題目中是否也包含該衝突群組的成員
            r_present = {member for member in group if member in r_lower}
            
            # 4. 如果 FAQ 題目中也有此群組的成員，但與用戶提問中的成員不同，則代表衝突。
            # （例如：用戶提問問的是「設計」專利，但 FAQ 題目是在回答「發明」專利，則兩者衝突不予匹配）
            if r_present and not (q_present & r_present):
                return True
                
    return False


def load_faq(csv_path: str) -> List[FaqRecord]:
    records: List[FaqRecord] = []
    if not os.path.exists(csv_path):
        logging.warning("FAQ file missing: %s", csv_path)
        return records

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            category = (
                row.get("\u985e\u5225")
                or row.get("\u985e\u578b")
                or row.get("category")
                or ""
            ).strip()
            question = (
                row.get("\u984c\u76ee")
                or row.get("\u554f\u984c")
                or row.get("question")
                or ""
            ).strip()
            answer = (
                row.get("\u89e3\u7b54")
                or row.get("\u7b54\u6848")
                or row.get("answer")
                or ""
            ).strip()
            updated = (row.get("\u66f4\u65b0\u65e5\u671f") or row.get("updated") or "").strip()
            record_id = (row.get("\u9805\u6b21") or str(index)).strip()
            synonyms = (
                row.get("\u5ef6\u4f38\u554f\u984c")
                or row.get("synonyms")
                or ""
            ).strip()

            if not question or not answer:
                continue

            search_text = normalize_text(f"{question} {synonyms}" if synonyms else question)
            records.append(
                FaqRecord(
                    record_id=record_id,
                    category=category,
                    question=question,
                    answer=answer,
                    updated=updated,
                    search_text=search_text,
                    synonyms=synonyms,
                )
            )

    return records

def build_messages(system_instruction: str, user_prompt: str) -> List[Dict[str, str]]:
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})
    return messages


def get_message_text(response) -> str:
    if not response or not getattr(response, "choices", None):
        return ""
    message = response.choices[0].message
    return message.content or ""


def llm_structured(
    client: Optional[OpenAI],
    model: str,
    prompt: str,
    response_schema: type[BaseModel],
    *,
    max_output_tokens: int = 512,
    temperature: float = 0.2,
    retries: int = 2,
    log_tag: str = "llm",
) -> Optional[BaseModel]:
    if not client:
        return None

    messages = build_messages(
        "你是一個專業的結構化輸出助手。請嚴格遵守提供的 Schema 格式進行輸出。",
        prompt,
    )
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "response_schema",
            "strict": True,
            "schema": response_schema.model_json_schema(),
        },
    }

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature if attempt == 0 else 0.0,
                max_tokens=max_output_tokens,
                response_format=response_format,
                seed=42,
                user="default_user",
            )

            text = get_message_text(response)
            if not text:
                logging.warning("Structured output response empty (%s)", log_tag)
                continue
            try:
                cleaned_text = text.replace('\xa0', ' ').strip()
                parsed = response_schema.model_validate_json(cleaned_text)
                logging.info("LLM response (%s) parsed successfully", log_tag)
                return parsed
            except Exception:
                logging.exception("Structured output parsing failed (%s): %s", log_tag, text)

        except Exception:
            logging.exception("LLM request failed (attempt %d)", attempt + 1)
            continue

    logging.error("LLM failed to return valid structured output after retries")
    return None


def check_coverage(query: str, best_record: "FaqRecord", client: Optional[OpenAI], model: str) -> bool:
    """判斷 reranker 選出的最佳 FAQ 條目，是否能單獨且完整地回答用戶的全部疑問。"""
    if not client:
        return True  # 無 LLM 時保守預設：不觸發 RAG
    prompt = (
        """你是專利問答系統的「覆蓋判斷員」暨「RAG 守門員」。
以下是系統檢索出最相關的單一 FAQ 條目。

### 💡 背景定位說明
1. 用戶提問：用戶針對專利業務提出的具體實務問題。
2. FAQ 條目：來自「中華民國智慧財產局專利常見問答 FAQ 文件」。
你的核心任務是把關：**是否真的需要觸發耗時且發散的 RAG 流程？** 因為前方的檢索系統已經選出置信度最高的條目，若該 FAQ 的「答案」核心資訊已經足夠解答用戶提問，你必須判定為覆蓋 (has_coverage = true)，讓系統直接回傳該 FAQ，避免濫用 RAG。

### 🚨 判斷覆蓋度之核心準則（極重要）
1. **口語修飾詞的寬容判定（防止過度嚴苛）**：用戶常在提問中加入日常修飾性定語或副詞（如「具體是」、「最晚」、「哪些」、「到底」等）來加強語氣。只要 FAQ 答案的核心內容或所包含 the 法條規定，實質上已經涵蓋了該修飾詞所指向的實務資訊，就必須判定為 `has_coverage = true`。
   - *例如*：用戶問「最晚什麼時候舉發」，FAQ 答案回「任何人得於專利權期間內提起舉發」— 雖然答案沒有使用「最晚」字眼，但其給出的時間期限「專利權期間內」已經實質回答了最晚時間，必須判定為 `has_coverage = true`。
   - *例如*：用戶問「實體審查是審查哪些要件」，FAQ 答案已簡述了實體審查的定義與對象，只要整體回答內容對應該程序，就判定為 `has_coverage = true`，不可因沒有逐一條列所有要件就判定不覆蓋。
2. **定義與條件的互相涵蓋**：
   - 用戶問「某專利定義」→ FAQ 題目是「何謂某專利？」→ has_coverage = true。
   - 用戶問「某程序在什麼情況下可以申請？」→ FAQ 題目是「何謂某程序？」或「申請程序為何？」→ 只要答案內含適用條件，has_coverage = true。
   - 用戶問「A與B的不同」→ FAQ 題目是「何謂A？」→ 若答案中有對比B，has_coverage = true；若無對比B，has_coverage = false。
3. **複合提問 (Multi-part Query) 的覆蓋判定**：
   - 若用戶一口氣詢問多個獨立問題（如：同時問時間與規費），請在 `is_multi_part_query` 標記為 true。
   - **注意**：複合提問不代表一定無法覆蓋！如果該 FAQ 條目的答案，**剛好完整解答了用戶提出的「所有」子問題**，`has_coverage` 依然必須為 true。
   - 只有當用戶的問題包含 A 與 B 兩部分，但 FAQ 僅回答了 A 而遺漏 B 時，`has_coverage` 才判定為 false（此時系統才會啟動 RAG 整合其他知識）。

### 輸出思考程序 (Chain-of-Thought)
在生成 JSON 結構時，請務必先在 `reasoning_process` 欄位以 1-2 句話寫出語意對比過程：分析用戶問題的「核心法律意圖 / 所有子問題」，並對照 FAQ 解答是否已無死角地涵蓋所有疑問。"""
        f"\n用戶提問：{query}"
        f"\nFAQ 問題：{best_record.question}"
        f"\nFAQ 答案：{best_record.answer}"
    )
    res = llm_structured(client, model, prompt, CoverageOutput, max_output_tokens=256, temperature=0.0, log_tag="coverage")
    if res:
        result = res.has_coverage
        logging.info("[Coverage CoT] Query: '%s' | Reasoning: %s | IsMultiPart: %s | has_coverage: %s", 
                     query, res.reasoning_process, res.is_multi_part_query, result)
    else:
        result = True
        logging.info("[Coverage CoT] LLM returned empty response. Fallback to has_coverage = True")
    return result


def get_embeddings(texts: List[str], client: Optional[OpenAI] = None, task_type: str = "") -> List[List[float]]:
    """
    取得文本的向量表示（embeddings）。
    說明：呼叫 LM Studio 的 embeddings 接口，忽略 task_type 參數。
    """
    if not texts:
        return []
    if not client:
        logging.warning("Embeddings client not available")
        return []
    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]
    except Exception:
        logging.exception("Failed to get embeddings from LM Studio")
        return []


def answer_from_general_knowledge(query: str, client: Optional[OpenAI], model: str) -> str:
    if not client:
        return "無法回答：未配置 LLM 客戶端。"

    prompt = f"請根據您的通用專利知識回答以下關於專利的實務問題。請提供詳細的說明與指引，並給予實質的專業建議。\n\n用戶提問：{query}"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=build_messages(
                "你是一個專業的智慧財產權與專利實務專家。請以通用專利知識解答用戶的提問，並給予具體可行的步驟或建議。請用繁體中文回答，請勿使用 Markdown 的標題語法（如 # 等），請用一般文字段落與條列式說明。",
                prompt,
            ),
            temperature=0.7,
            max_tokens=1024,
            seed=42,
            user="default_user",
        )
        text = get_message_text(response)
        return text or "LLM 未能生成有效回答。"
    except Exception:
        logging.exception("Failed to get general knowledge answer")
        return "獲取通用知識回答時發生異常錯誤。"


def synthesize_answer_from_vector(
    query: str,
    top_candidates: List[Tuple[FaqRecord, float]],
    client: Optional[OpenAI],
    model: str,
) -> Tuple[str, List[str]]:
    if not client:
        return "無法回答：未配置 LLM 客戶端。", [rec.record_id for rec, _ in top_candidates]

    # 建立向量檢索的上下文內容
    context_parts = []
    for idx, (rec, score) in enumerate(top_candidates, start=1):
        context_parts.append(
            f"檢索到的 FAQ 條目 {idx} (相似度分數: {score:.4f})\n"
            f"條目ID: {rec.record_id}\n"
            f"類別: {rec.category}\n"
            f"問題: {rec.question}\n"
            f"答案: {rec.answer}\n"
        )
    context = "\n---\n".join(context_parts)

    system_instruction = (
        "你是一個專業且具有親和力的智慧財產局專利諮詢顧問。\n"
        "請使用提供給你的相關 FAQ 條目作為知識庫，在接下來的回答中，針對用戶提出的專利問題，進行結構化且條理清晰的答覆。\n"
        "請你以這些 FAQ 條目為基礎，提煉出最核心的實務資訊，以解答用戶的核心疑問。切勿編造任何條目中不存在的官方規費、期限或法律效力，以避免給予用戶錯誤的行政指引，這在專利實務上是非常嚴重的疏失。\n"
        "請以簡明扼要、易於閱讀的排版輸出。你可以使用基本的 Markdown 語法（例如 *, **, #, -, >, `, ``` 等）來美化回答，並在回答中適當加上引用標記 `[#ID]`，其中 ID 是 FAQ 條目的 record_id（例如：`[#12]`）。\n"
        "遵守以下規則：\n"
        "1. 必須嚴格根據提供的 FAQ 條目內容來回答，如果 FAQ 內容不足以回答用戶問題，請明確告知用戶，切勿自己編造。如果 FAQ 中有提到官方的規費或程序，請以 FAQ 為準，不可憑空臆測。\n"
        "2. 答覆結構規範：答覆必須簡明扼要。請先以一句話簡述回答，再依序詳細說明 FAQ 檢索庫中對應條目的詳細實務程序、規費與做法。\n"
        "3. **引用格式規定（極重要，必須嚴格遵守）：**\n"
        "   - 當你的回答中引用了某個 FAQ 條目的內容時，請在該段文字或句子後方，加上對應的引用標記 `[#ID]`。其中 ID 是 FAQ 條目的 record_id（例如：`[#12]`）。如果引用了多個條目，請分別標記，例如 `[#2] [#261]`。\n"
        "   - **引用格式不合格 of 例子**：禁止使用 `[FAQ 條目 X]`、`[條目 X]`、`[X]` 或 `[FAQ X]` 等非標準格式。此類不合格格式會直接導致系統解析失敗。\n"
        "   - **正確與錯誤範例**：\n"
        "     * ❌ 錯誤引用格式：根據[FAQ 2]的說明...\n"
        "     * ❌ 錯誤引用格式：專利法第 261 條規定，申請人可以...[2]\n"
        "     * ✅ 正確引用格式：專利申請案如果需要委任代理人，應檢附委任書...[#2]，且應由...[#261]\n"
        "4. **訊息一致性與去衝突過濾**：若發現檢索到的多個 FAQ 條目在規費、期限等資訊上有互相衝突或不一致，請以最新更新日期的條目為準，並在回答中提醒用戶注意。\n"
        "5. **避免贅字與過度重複引用**：如果同一段話多次引用了同一個條目，只需要在段落結尾加上一次 `[#ID]` 即可，不需字字句句都標記。\n"
        "6. 回答一律使用繁體中文輸出，語氣必須保持專業、親切且有禮貌。\n"
        "7. 如果發現檢索結果與用戶的問題有部分不相符或有所落差，請在回答中適度提醒用戶，例如說明本回答是針對發明專利，而新型專利的規定可能有所不同，以免用戶產生誤解。\n"
    )

    prompt = (
        f"用戶提問：{query}\n\n"
        f"檢索到的 FAQ 參考內容如下：\n"
        f"{context}\n"
    )

    candidate_ids = [rec.record_id for rec, _ in top_candidates]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=build_messages(system_instruction, prompt),
            temperature=0.2,
            max_tokens=2048,
            seed=42,
            user="default_user",
        )
        text = get_message_text(response)
        if not text:
            return "LLM 未能成功合成解答。", candidate_ids

        # 1. 抓取引用的 ID 標記 [#ID]
        raw_ids = re.findall(r'\[#\s*(\w+)\]', text)
        if not raw_ids:
            raw_ids = re.findall(r'\[(?:FAQ|條目|蝚)?\s*#?\s*(\w+)\]', text)

        used_ids = []
        for r_id in raw_ids:
            r_id = r_id.strip()
            if r_id and r_id not in used_ids:
                used_ids.append(r_id)

        # 過濾以確保引用 ID 在候選名單中
        used_ids = [uid for uid in used_ids if uid in candidate_ids]

        if not used_ids:
            logging.warning("[RAG] No valid citation tags found in LLM response. Fallback to all candidates.")
            used_ids = candidate_ids

        # 移除回答中的引用標籤
        clean_text = re.sub(r'\s*\[(?:FAQ|條目|蝚)?\s*#?\s*\w+\]', '', text)
        return clean_text, used_ids
    except Exception:
        logging.exception("Failed to synthesize answer from vector context")
        return "合成解答時發生錯誤。", candidate_ids


class VectorSearcher:
    def __init__(self, records: List[FaqRecord], client: Optional[OpenAI], cache_path: str = "faq_embeddings_qonly.json"):
        self.records = records
        self.client = client
        self.cache_path = cache_path
        self.embeddings: Dict[str, List[float]] = {}
        if client and records:
            self._load_or_build_cache()

    def _load_or_build_cache(self):
        # 檢查快取檔案是否存在
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                
                # 檢查快取資料格式是否正確，並比對模型名稱是否一致
                if isinstance(cache_data, dict) and "model_name" in cache_data:
                    if cache_data.get("model_name") == EMBEDDING_MODEL:
                        self.embeddings = cache_data.get("embeddings", {})
                        # 檢查是否所有必要的 record_id_q 與 record_id_s 都已經有快取向量
                        all_cached = True
                        for r in self.records:
                            q_key = f"{r.record_id}_q"
                            s_key = f"{r.record_id}_s"
                            if q_key not in self.embeddings:
                                all_cached = False
                                break
                            if getattr(r, "synonyms", None) and s_key not in self.embeddings:
                                all_cached = False
                                break
                        if all_cached:
                            logging.info("Loaded FAQ embeddings from cache (%s).", EMBEDDING_MODEL)
                            return
                
                logging.warning("Cache version mismatch or incomplete. Rebuilding...")
            except Exception:
                logging.exception("Failed to load embeddings cache")

        logging.info("Building FAQ embeddings cache using %s...", EMBEDDING_MODEL)
        
        # [Embedding 技術限制標記 - 多向量最大相似度設計 (Multi-Vector Max Similarity)]
        # 由於 BGE-M3 等 Dense Embedding 模型在極短句（≤ 10 字，如「何謂新型？」）上的語義空間特徵較為稀疏，
        # 難以直接與使用者具體、長篇幅的口語或法律術語（如「新型專利具體的法律定義到底是什麼？」）進行精準匹配。
        # 同時，若直接字串拼接會造成「前綴稀釋效應」，反而拉低餘弦相似度。
        # 因此，我們將「原始題目」與「延伸問題」作為兩個獨立向量進行 Embedding 編碼，檢檢索時計算兩者相似度並取最大值 (MAX)。
        keys_to_embed = []
        texts_to_embed = []
        for r in self.records:
            keys_to_embed.append(f"{r.record_id}_q")
            texts_to_embed.append(r.question)
            if getattr(r, "synonyms", None):
                keys_to_embed.append(f"{r.record_id}_s")
                texts_to_embed.append(r.synonyms)

        all_values = get_embeddings(texts_to_embed, self.client)

        if len(all_values) == len(keys_to_embed):
            self.embeddings = {k: v for k, v in zip(keys_to_embed, all_values)}
            try:
                # 快取資料寫入檔案以供後續使用
                cache_to_save = {
                    "model_name": EMBEDDING_MODEL,
                    "embeddings": self.embeddings
                }

                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache_to_save, f)
                logging.info("Saved FAQ embeddings to cache.")
            except Exception:
                logging.exception("Failed to save embeddings cache")

    def search(self, query: str) -> Optional[Tuple[FaqRecord, float]]:
        if not self.embeddings:
            return None

        query_vecs = get_embeddings([query], self.client)
        if not query_vecs:
            return None
        query_vec = query_vecs[0]

        query_vec_tensor = torch.tensor(query_vec)
        
        best_record = None
        best_score = -1.0

        for record in self.records:
            q_key = f"{record.record_id}_q"
            s_key = f"{record.record_id}_s"
            
            score_q = -1.0
            score_s = -1.0
            
            if q_key in self.embeddings:
                vec_q = torch.tensor(self.embeddings[q_key])
                score_q = util.cos_sim(query_vec_tensor, vec_q).item()
                
            if getattr(record, "synonyms", None) and s_key in self.embeddings:
                vec_s = torch.tensor(self.embeddings[s_key])
                score_s = util.cos_sim(query_vec_tensor, vec_s).item()
                
            score = max(score_q, score_s)

            if score > best_score:
                best_score = float(score)
                best_record = record

        return (best_record, best_score) if best_record else None

    def search_top_k(self, query: str, top_k: int = 3) -> List[Tuple[FaqRecord, float]]:
        if not self.embeddings:
            return []

        query_vecs = get_embeddings([query], self.client)
        if not query_vecs:
            return []
        query_vec = query_vecs[0]

        query_vec_tensor = torch.tensor(query_vec)
        
        results = []
        for record in self.records:
            q_key = f"{record.record_id}_q"
            s_key = f"{record.record_id}_s"
            
            score_q = -1.0
            score_s = -1.0
            
            if q_key in self.embeddings:
                vec_q = torch.tensor(self.embeddings[q_key])
                score_q = util.cos_sim(query_vec_tensor, vec_q).item()
                
            if getattr(record, "synonyms", None) and s_key in self.embeddings:
                vec_s = torch.tensor(self.embeddings[s_key])
                score_s = util.cos_sim(query_vec_tensor, vec_s).item()
                
            best_score = max(score_q, score_s)
            results.append((record, float(best_score)))

        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]


class BM25Index:
    def __init__(self, records: List[FaqRecord], k1: float = 1.5, b: float = 0.75):
        self.records = records
        self.k1 = k1
        self.b = b
        self.record_ids = [r.record_id for r in records]
        self.doc_count = len(records)
        self.doc_len: Dict[str, int] = {}
        self.term_freqs: Dict[str, Counter] = {}
        self.doc_freq: Dict[str, int] = {}
        self.avgdl = 0.0
        self._build()

    def matches(self, records: List[FaqRecord]) -> bool:
        if len(records) != len(self.record_ids):
            return False
        return self.record_ids == [r.record_id for r in records]

    def _build(self) -> None:
        total_len = 0
        for record in self.records:
            # 合併題目與延伸問題以建立更健壯的 BM25 詞彙索引
            text_to_tokenize = record.question
            if getattr(record, "synonyms", None):
                text_to_tokenize = f"{record.question} {record.synonyms}"
            tokens = tokenize_bm25(text_to_tokenize)
            freq = Counter(tokens)
            self.term_freqs[record.record_id] = freq
            doc_len = sum(freq.values())
            self.doc_len[record.record_id] = doc_len
            total_len += doc_len
            for term in freq.keys():
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

        self.avgdl = total_len / max(self.doc_count, 1)

    def score(self, query: str) -> Dict[str, float]:
        tokens = tokenize_bm25(query)
        if not tokens or self.doc_count == 0:
            return {}

        query_terms = Counter(tokens)
        scores: Dict[str, float] = {}
        avgdl = self.avgdl or 1.0

        for record_id, tf in self.term_freqs.items():
            dl = self.doc_len.get(record_id, 0)
            score = 0.0
            for term in query_terms.keys():
                df = self.doc_freq.get(term, 0)
                if df == 0:
                    continue
                idf = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                score += idf * (freq * (self.k1 + 1.0) / denom)

            if score > 0.0:
                scores[record_id] = score

        return scores


def fuzzy_similarity(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    return (
        0.3 * fuzz.WRatio(text_a, text_b)
        + 0.25 * fuzz.token_set_ratio(text_a, text_b)
        + 0.15 * fuzz.partial_ratio(text_a, text_b)
        + 0.3 * fuzz.partial_token_sort_ratio(text_a, text_b)
    )


def keyword_similarity(keywords: List[str], text: str) -> float:
    if not keywords:
        return 0.0
    scores = [fuzz.WRatio(keyword, text) for keyword in keywords if keyword]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def keyword_hit_count(keywords: List[str], text: str) -> int:
    if not keywords:
        return 0
    hits = 0
    for keyword in keywords:
        if keyword and keyword in text:
            hits += 1
    return min(hits, 10)


def normalize_keywords(keywords: List[str]) -> List[str]:
    cleaned: List[str] = []
    for keyword in keywords:
        if not isinstance(keyword, str):
            continue
        normalized = normalize_text(keyword)
        if normalized:
            cleaned.append(normalized)
    return cleaned


def extract_keywords(
    query: str,
    client: Optional[OpenAI],
    model: str,
) -> Tuple[List[str], List[str], bool]:
    prompt = ("""你是一個專門處理繁體中文專利 FAQ 系統的關鍵字提取專家。
你的任務是分析用戶查詢，提取核心專利實務概念，以供資料庫進行精確比對。

## 背景知識：專利基本術語
- 審查方案與管道：PPH (專利審查高速公路)、AEP (專利事由加速審查)、TW-SUPA、一般審查。
- 專利類型：發明專利、新型專利、設計專利。
- 申請主體與關係人：申請人、發明人、代理人、專利代理人、專利師、自然人、法人。
- 規費與程序：申請規費、年費、減免、退費、委任書。
- 文件與審查：說明書、申請書、摘要、圖式、實體審查、新型技術報告、補正、申復、答辯、面詢、優先權、分割。

## 關鍵字提取規則
1. Primary Keywords (主要關鍵字)：核心專利術語、類型與程序名稱（如：發明專利、規費減免、實體審查、AEP）。此類別需具備高度檢索區分度。
2. Secondary Keywords (次要關鍵字)：輔助說明屬性、時間或補充概念（如：期限、外文、金額、應檢附文件）。
3. 嚴格過濾無意義詞彙：絕對禁止提取「請問、我想知道、有沒有、是不是、可以嗎、怎麼辦、該如何、為什麼、何謂、什麼是」等口語問句或助詞。
4. 術語正規化 (同義轉換)：將口語詞彙精確對應至正式專利術語。
   - 「加速審查/快速審查」 -> 提取為「AEP」或「PPH」。
   - 「改名字/換人」 -> 依據上下文判斷，提取為「變更申請人」、「變更發明人」或「變更代理人」。
   - 「省錢/變便宜」 -> 提取為「規費減免」。

## 輸出格式規範
- 每一類關鍵字最多提取 5 個。
- 僅能輸出純 JSON 格式，嚴禁使用 Markdown 標記（如 ```json）、嚴禁加入任何解釋性文字。
- 必須嚴格符合以下 JSON 結構：
{"primary_keywords": ["詞1", "詞2"], "secondary_keywords": ["詞3", "詞4"]}

## 提取範例

【範例 1：定義型問題】
用戶提問：新型專利具體的法律定義到底是什麼？
預期輸出：{"primary_keywords": ["新型專利"], "secondary_keywords": ["定義", "法律"]}

【範例 2：時間型問題】
用戶提問：專利權人可以在什麼時候向智慧局申請專利權更正？
預期輸出：{"primary_keywords": ["專利權更正"], "secondary_keywords": ["時間", "申請", "何時"]}

【範例 3：口語與術語正規化】
用戶提問：如果不找人代辦，自己辦理發明專利，規費會不會比較便宜？
預期輸出：{"primary_keywords": ["發明專利", "規費減免"], "secondary_keywords": ["自辦", "規費"]} """
        + f"用戶提問：{query}\n"
    )

    data = llm_structured(
        client,
        model,
        prompt,
        KeywordOutput,
        max_output_tokens=512,
        temperature=0,
        log_tag="keywords",
    )
    if not data:
        return [], [], False

    primary = normalize_keywords(data.primary_keywords)[:5]
    secondary = normalize_keywords(data.secondary_keywords)[:5]
    has_keywords = bool(primary or secondary)
    return primary, secondary, has_keywords


def rerank_candidates(
    query: str,
    candidates: List[Dict],
    client: Optional[OpenAI],
    model: str,
) -> Optional[Dict[str, float]]:
    if not candidates:
        return None

    logging.info("Reranking %d candidates for query: %s", len(candidates), query)

    entries = []
    for idx, item in enumerate(candidates, start=1):
        record = item["record"]
        entries.append(
            {
                "id": record.record_id,
                "question": record.question,
                "answer": record.answer,
            }
        )

    entries_json = json.dumps(entries, ensure_ascii=False)
    prompt = ("""你是一個專業的智慧財產局專利諮詢 Rerank 專家。你的任務是評估用戶提出的查詢與系統檢索出的 FAQ 候選條目之間的相關性分數。
請仔細閱讀用戶的問題 (query) 以及每一個候選條目的問題 (question) 與答案 (answer)。

### 評估基準
- question：FAQ 候選條目的問題
- answer：FAQ 候選條目的答案

### 評分細則與優先準則
評分時請秉持以下原則：
1. **第一優先：主體與法律概念判定**：評估用戶提問與候選條目的 question 是否存在法律概念或主體上的本質衝突。若主題不符或存在衝突，觸發一票否決，分數必須低於或等於 30 分。
2. **第二優先：解答實質度判定**：評估候選條目的 answer 是否能實質解答用戶問題的核心實務疑問。

## 評分級別說明
- 90-100分 (極度相關)：用戶提問與 FAQ 問題在法律意圖上完全一致，且答案能直接完美解答。
- 80-89分 (高度相關)：用戶問題與 FAQ 問題字面同義改寫，答案提供核心解答，僅有極微細微的描述差異。
- 70-79分 (中度相關)：主題基本相符，但用戶問題包含一些細節在 FAQ 中沒有完全對應，但答案依然有很高的實用參考價值。
- 40-69分 (低度相關)：主題勉強相關，但答案無法直接解答用戶的核心疑問（例如問期限但答案只列出規費）。
- 0-30分 (不相關/衝突/完全無關)：概念有衝突或完全沒有任何參考價值。

## 嚴格一票否決準則（極重要，必須嚴格遵守）
若遇到以下任一衝突或不符，該條目判定為完全不相關，**分數一律不得超過 30 分**（除特殊焦點衝突另有規定外）：
1. **主體關係人衝突**：例如用戶提問是問「申請人」，但候選條目是關於「發明人」或「代理人」的變更或義務。
2. **專利類型衝突**：例如用戶提問是關於「發明專利」，但候選條目是關於「新型專利」（如新型技術報告）或「設計專利」。
3. **加速審查方案衝突**：例如用戶提問是問「AEP」，但候選條目是關於「PPH」或「TW-SUPA」等互斥管道。
4. **法定程序與概念衝突**：例如用戶問「更正」，選項是「補換發專利證書」；或者用戶問「延緩實體審查」，選項是「專利權期間延長」。這類在法律上為完全獨立的行政程序，必須嚴格區分。
5. **提問焦點衝突**：如果用戶問的是「申請的時間點或期限」，但選項是在回答「申請的條件或資格」（或反之），焦點完全不同，視為不相關。
6. **列舉與比較焦點衝突**：當用戶問的是「種類有哪些 / 有幾種」（列舉型問題，例如：有幾種專利類型），而候選 FAQ 回答的是這些專利的「不同點 / 有何不同」（比較型問題，例如：發明與新型有何不同），或者反之。這兩者雖然關鍵字高度重疊，但提問焦點完全不同，回答後者無法直接解答前者的列舉問題。此情況下，該候選 FAQ 分數一律不得超過 65 分。

## 評分範例說明

### 範例 1：主體關係人衝突一票否決
- 用戶提問：「發明人是不是可以改名字或變更發明人？」
- 候選 FAQ：[問題] 「申請人姓名或名稱變更，應檢附什麼文件？」
- **評估與思考程序 (reasoning)**：用戶提問核心是「發明人」的變更，而候選 FAQ 是回答「申請人」的姓名變更。在專利實務與法律程序中，「發明人」與「申請人」是完全不同的主體，行政程序完全不同。觸發主體關係人衝突之一票否決。
- **評分 (score)**：30

### 範例 2：加速審查方案衝突一票否決
- 用戶提問：「我們想申請 AEP，應該怎麼做？」
- 候選 FAQ：[問題] 「申請專利審查高速公路(PPH)應符合什麼條件？」
- **評估與思考程序 (reasoning)**：用戶提問要求申請 AEP (專利事由加速審查)，而候選 FAQ 是介紹 PPH (專利審查高速公路)。兩者是完全不同且互斥的加速審查管道。觸發加速方案衝突之一票否決。
- **評分 (score)**：20

### 範例 3：提問焦點衝突（時間 vs 條件）
- 用戶提問：「我們可以向 AEP 系統提出申請的時間點是什麼時候？」
- 候選 FAQ：[問題] 「申請 AEP 應具備什麼條件？」
- **評估與思考程序 (reasoning)**：用戶問的是申請的「時間點(何時)」，候選 FAQ 回答的是申請的「條件(資格)」。時間與條件是完全不同的實務焦點，回答條件並不能解開用戶對時間的疑問。觸發提問焦點衝突之一票否決。
- **評分 (score)**：30

### 範例 4：定義涵蓋實務（完美契合）
- 用戶提問：「專利法所稱的誤譯訂正，這在什麼情況下可以向智慧局申請？」
- 候選 FAQ：[問題] 「何謂誤譯訂正？」
- **評估與思考程序 (reasoning)**：用戶詢問「在什麼情況下可以申請」，候選 FAQ 題目是「何謂誤譯訂正」。在官方 FAQ 中，「何謂某程序」的答案通常會完整涵蓋該程序的定義、適用情況與申請時機。無任何衝突，高度契合。
- **評分 (score)**：95

### 範例 5：列舉與比較焦點衝突
- 用戶提問：「台灣有哪幾種不同的專利類型？」
- 候選 FAQ A：[問題] 「我國的專利種類有幾種？」
  - **評估與思考程序 (reasoning)**：用戶問的是台灣專利的種類有哪些，FAQ A 的問題完全匹配此列舉焦點，且答案能直接給出解答，屬於極度相關。
  - **評分 (score)**：95
- 候選 FAQ B：[問題] 「發明、新型及設計專利有何不同？」
  - **評估與思考程序 (reasoning)**：用戶問的是有哪些種類（列舉型），而 FAQ B 是在比較三種專利的不同點（比較型）。雖然兩者都包含「發明、新型、設計」等關鍵字，但提問焦點不同，FAQ B 無法直接列舉專利種類。屬於中低度相關。
  - **評分 (score)**：65

## 輸出規範
- 請直接輸出符合系統定義之 JSON Schema 的結構化資料，嚴禁使用任何 Markdown 標記（如 ```json）。
- 必須針對清單中每個條目的 id，先在 `reasoning` 欄位寫下評評估程序，再給出 `score`。
"""
        f"\n用戶提問：{query}\n"
        f"\n候選 FAQ 列表 (JSON 格式)：\n"
        f"{entries_json}\n"
    )

    data = llm_structured(
        client,
        model,
        prompt,
        RerankOutput,
        max_output_tokens=4096,
        temperature=0.0,
        log_tag="rerank",
    )
    if not data:
        return None

    scores = {item.id: item.score for item in data.scores}
    logging.info("Rerank scores: %s", scores)
    return scores


def compute_candidates(
    query: str,
    records: List[FaqRecord],
    client: Optional[OpenAI],
    model: str,
) -> Tuple[List[Dict], List[str], List[str]]:
    query_norm = normalize_text(query)
    global BM25_INDEX
    bm25_scores: Dict[str, float] = {}
    bm25_ranks: Dict[str, int] = {}
    bm25_rank_scores: Dict[str, float] = {}
    if records:
        if BM25_INDEX is None or not BM25_INDEX.matches(records):
            BM25_INDEX = BM25Index(records)
        bm25_scores = BM25_INDEX.score(query)
        if bm25_scores:
            # 只有當 BM25 原始分數顯著大於等於 0.5 時，才列入排名計算，過濾低分噪聲
            valid_scores = {rid: sc for rid, sc in bm25_scores.items() if sc >= 0.5}
            if valid_scores:
                sorted_bm25 = sorted(valid_scores.items(), key=lambda item: item[1], reverse=True)
                rank_scale = max(len(sorted_bm25) - 1, 1)
                for rank, (record_id, score) in enumerate(sorted_bm25, start=1):
                    bm25_ranks[record_id] = rank
                    bm25_rank_scores[record_id] = 100.0 - (rank - 1) * (100.0 / rank_scale)
    primary_keywords, secondary_keywords, keyword_ready = extract_keywords(query, client, model)
    all_keywords = primary_keywords + secondary_keywords

    # 計算向量檢索相似度 (使用多向量最大相似度設計)
    vector_scores: Dict[str, float] = {}
    if VECTOR_SEARCHER and VECTOR_SEARCHER.embeddings:
        query_vecs = get_embeddings([query], client)
        if query_vecs:
            query_vec_tensor = torch.tensor(query_vecs[0])
            for record in records:
                q_key = f"{record.record_id}_q"
                s_key = f"{record.record_id}_s"
                
                score_q = -1.0
                score_s = -1.0
                
                if q_key in VECTOR_SEARCHER.embeddings:
                    vec_tensor = torch.tensor(VECTOR_SEARCHER.embeddings[q_key])
                    score_q = util.cos_sim(query_vec_tensor, vec_tensor).item()
                    
                if getattr(record, "synonyms", None) and s_key in VECTOR_SEARCHER.embeddings:
                    vec_tensor_s = torch.tensor(VECTOR_SEARCHER.embeddings[s_key])
                    score_s = util.cos_sim(query_vec_tensor, vec_tensor_s).item()
                    
                best_score = max(score_q, score_s)
                vector_scores[record.record_id] = float(best_score) * 100.0

    candidates: List[Dict] = []
    for record in records:
        base_text = normalize_text(record.question)
        base_score = fuzzy_similarity(query_norm, base_text)
        if getattr(record, "synonyms", None):
            # 對於延伸同義問題，計算 query 與每個同義句的 fuzzy ratio，取最大值以防止短題目被嚴重壓分
            synonym_list = [s.strip() for s in record.synonyms.split("；") if s.strip()]
            for syn in synonym_list:
                syn_score = fuzzy_similarity(query_norm, normalize_text(syn))
                if syn_score > base_score:
                    base_score = syn_score
        vec_score = vector_scores.get(record.record_id, 0.0)
        bm25_score = bm25_scores.get(record.record_id, 0.0)
        bm25_rank = bm25_ranks.get(record.record_id)
        bm25_rank_score = bm25_rank_scores.get(record.record_id, 0.0)

        # 長度與字元重疊補償 (Length & Character Overlap Compensation)
        # 對於較短的 FAQ 題目（例如字數 <= 6），若字元重疊比例 >= 50%，則給予適當的相似度補償。
        # 這是為了解決短關鍵字在向量檢索中相似度容易偏低，但實際上卻是精確匹配的問題。
        q_len = len(record.question)
        length_compensation = 0.0
        if q_len <= 6:
            q_chars = set(c for c in query if c.isalnum())
            faq_chars = set(c for c in record.question if c.isalnum())
            if faq_chars:
                overlap_ratio = len(faq_chars & q_chars) / len(faq_chars)
                if overlap_ratio >= 0.5:
                    length_compensation = (8.0 - q_len) * 1.5

        if keyword_ready:
            primary_score = keyword_similarity(primary_keywords, record.search_text)
            secondary_score = keyword_similarity(secondary_keywords, record.search_text)
            hit_count = keyword_hit_count(all_keywords, record.search_text)
            
            # 主要關鍵字在題目中的精確命中次數加成 (Primary Keyword Exact Hits Boost)
            primary_exact_hits = sum(1 for kw in primary_keywords if kw and kw in record.question)
            
            # 計算綜合檢索評分
            total_score = (
                0.25 * base_score
                + 0.45 * vec_score
                + 0.25 * primary_score
                + 0.05 * secondary_score
                + 0.12 * bm25_rank_score
                + hit_count
                + (primary_exact_hits * 3.0)
                + length_compensation
            )
        else:
            primary_score = None
            secondary_score = None
            hit_count = None
            total_score = 0.30 * base_score + 0.70 * vec_score + 0.15 * bm25_rank_score + length_compensation

        candidates.append(
            {
                "record": record,
                "base_score": base_score,
                "vector_score": vec_score,
                "bm25_score": bm25_score,
                "bm25_rank": bm25_rank,
                "bm25_rank_score": bm25_rank_score,
                "primary_score": primary_score,
                "secondary_score": secondary_score,
                "hit_count": hit_count,
                "total_score": total_score,
                "keyword_ready": keyword_ready,
                "length_compensation": length_compensation,
            }
        )

    candidates.sort(key=lambda item: item["total_score"], reverse=True)
    return candidates, primary_keywords, secondary_keywords


def format_answer(record: FaqRecord, low_confidence: bool) -> str:
    lines = [
        f"Category: {record.category}",
        f"Question: {record.question}",
        "",
        record.answer,
    ]
    return "\n".join(lines).strip()


def build_debug_rows(
    query: str,
    candidates: List[Dict],
    best_id: str,
    llm_scores: Optional[Dict[str, float]],
    primary_keywords: List[str],
    secondary_keywords: List[str],
) -> List[Dict]:
    timestamp = datetime.now(timezone.utc).isoformat()
    rows: List[Dict] = []

    def format_score(value: Optional[float]) -> Union[str, float]:
        if value is None:
            return ""
        return round(value, 4)

    def format_keywords(values: List[str]) -> str:
        if not values:
            return ""
        return ", ".join(values)

    primary_text = format_keywords(primary_keywords)
    secondary_text = format_keywords(secondary_keywords)

    for rank, item in enumerate(candidates, start=1):
        record = item["record"]
        llm_score = ""
        if llm_scores and record.record_id in llm_scores:
            llm_score = round(llm_scores[record.record_id], 4)
        rows.append(
            {
                "timestamp": timestamp,
                "query": query,
                "stage": "final",
                "rank": rank,
                "record_id": record.record_id,
                "category": record.category,
                "question": record.question,
                "primary_keywords": primary_text,
                "secondary_keywords": secondary_text,
                "base_score": round(item.get("base_score", 0.0), 4) if item.get("base_score") is not None else "",
                "vector_score": round(item.get("vector_score", 0.0), 4) if item.get("vector_score") is not None else "",
                "primary_score": format_score(item.get("primary_score")),
                "secondary_score": format_score(item.get("secondary_score")),
                "hit_count": "" if item.get("hit_count") is None else item.get("hit_count"),
                "total_score": round(item.get("total_score", 0.0), 4),
                "llm_score": llm_score,
                "final_score": round(item.get("final_score", 0.0), 4),
                "selected": "yes" if record.record_id == best_id else "",
            }
        )

    return rows


def append_debug_rows(rows: List[Dict], csv_path: str) -> None:
    if not rows or not csv_path:
        return

    columns = [
        "timestamp",
        "query",
        "stage",
        "rank",
        "record_id",
        "category",
        "question",
        "primary_keywords",
        "secondary_keywords",
        "base_score",
        "vector_score",
        "primary_score",
        "secondary_score",
        "hit_count",
        "total_score",
        "llm_score",
        "final_score",
        "selected",
    ]

    directory = os.path.dirname(csv_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def answer_query(
    query: str,
    records: List[FaqRecord],
    client: Optional[OpenAI],
    model: str,
) -> Dict:
    # 說明：低置信度閾值 (60.0) 是結合檢索相似度與 LLM 評估分數的綜合評定基準。
    # 若綜合評分低於此值，則代表檢索結果不夠精確，系統將啟動 RAG 或通用知識回答機制。
    low_confidence_threshold = 60.0
    top_k = 12
    score_gap_threshold = 8.0

    # 初始化除錯輸出參數
    exact_match = "N/A"
    fuzzy_match = "N/A"
    semantic_match = "N/A"
    score_gap_ok = "N/A"
    has_conflict = "N/A"
    primary_keywords_str = "N/A"
    secondary_keywords_str = "N/A"
    top2_question = "N/A"
    top2_score = "N/A"
    has_coverage_val = "N/A"
    candidates_debug = []

    if query:
        # 清除查詢字串中的不間斷空格 (\xa0) 與全形空格 (\u3000)，以避免影響 LLM 解析與 JSON 輸出格式。
        query = query.replace('\xa0', ' ').replace('\u3000', ' ').strip()

    if not query or not query.strip():
        return {
            "answer": "Please enter a question.",
            "full_answer": "Please enter a question.",
            "category": "",
            "question": "",
            "score": 0.0,
            "low_confidence": True,
            "response_type": "standard",
        }

    if not records:
        return {
            "answer": "FAQ data not available.",
            "full_answer": "FAQ data not available.",
            "category": "",
            "question": "",
            "score": 0.0,
            "low_confidence": True,
            "response_type": "standard",
        }

    candidates, primary_keywords, secondary_keywords = compute_candidates(
        query,
        records,
        client,
        model,
    )

    if candidates:
        if primary_keywords:
            primary_keywords_str = ", ".join(primary_keywords)
        if secondary_keywords:
            secondary_keywords_str = ", ".join(secondary_keywords)
        if len(candidates) > 1:
            top2_question = candidates[1]["record"].question
            top2_score = round(candidates[1]["total_score"], 2)

        for idx, item in enumerate(candidates[:8]):
            candidates_debug.append({
                "record_id": item["record"].record_id,
                "question": item["record"].question,
                "total_score": round(item.get("total_score", 0.0), 2),
                "llm_score": "N/A",
                "final_score": round(item.get("total_score", 0.0), 2),
                "rank": idx + 1
            })

        query_norm = normalize_text(query)
        top_candidate = candidates[0]
        second_score = candidates[1]["total_score"] if len(candidates) > 1 else None
        score_gap_ok = (
            True
            if second_score is None
            else (top_candidate["total_score"] - second_score) >= score_gap_threshold
        )
        top_question_norm = normalize_text(top_candidate["record"].question)
        exact_match = bool(query_norm and top_question_norm and query_norm == top_question_norm)
        fuzzy_match = False
        if not exact_match and query_norm and top_question_norm:
            fuzzy_match = fuzzy_similarity(query_norm, top_question_norm) >= 88.0

        # 定義語意匹配 (Semantic Match) 的高置信度判定基準
        # 使用 text-embedding-bge-m3 模型時，若檢索向量分數 >= 85.0 且與第二名的差距足夠大，則視為語意匹配。
        semantic_match = False
        if not exact_match and not fuzzy_match:
            if top_candidate["vector_score"] >= 88.0 and score_gap_ok:
                semantic_match = True

        bm25_match = False
        bm25_gap_ok = False
        bm25_rank = top_candidate.get("bm25_rank")
        bm25_top = top_candidate.get("bm25_score", 0.0)
        bm25_second = candidates[1].get("bm25_score", 0.0) if len(candidates) > 1 else None
        if bm25_rank == 1 and bm25_top >= 1.5:
            if bm25_second is None:
                bm25_gap_ok = True
            else:
                bm25_gap_ok = bm25_top >= (bm25_second + 0.001) * 1.5
            if bm25_gap_ok:
                bm25_match = True

        has_conflict = check_conflict(query, top_candidate["record"].question)

        # 優化快速鎖定防禦網，若第二名分數 >= 75.0 (雙峰警戒水位)，不允許語意/詞頻 bypass，強制送 Reranker
        if not has_conflict and (score_gap_ok and (exact_match or fuzzy_match or 
            (semantic_match and (second_score is None or second_score < 75.0)) or 
            (bm25_match and (second_score is None or second_score < 75.0)))):
            for item in candidates:
                item["final_score"] = item["total_score"]

            best = candidates[0]
            low_confidence = best["final_score"] < low_confidence_threshold
            debug_rows = build_debug_rows(
                query,
                candidates,
                best["record"].record_id,
                None,
                primary_keywords,
                secondary_keywords,
            )
            append_debug_rows(debug_rows, DEBUG_CSV_PATH)

            logging.info(
                "[Guardrail] Exact/near match locked (response_type: standard): %s (Score: %.2f)",
                best["record"].question,
                best["final_score"],
            )
            return {
                "answer": best["record"].answer,
                "full_answer": format_answer(best["record"], low_confidence),
                "category": best["record"].category,
                "question": best["record"].question,
                "record_id": best["record"].record_id,
                "score": round(best["final_score"], 2),
                "low_confidence": low_confidence,
                "response_type": "standard",
                "retrieval_score": round(best["total_score"], 2),
                "retrieval_rank": 1,
                "llm_score": "N/A",
                "llm_rank": "N/A",
                "final_score": round(best["final_score"], 2),
                "final_rank": 1,
                # 除錯元數據
                "exact_match": exact_match,
                "fuzzy_match": fuzzy_match,
                "semantic_match": semantic_match,
                "score_gap_ok": score_gap_ok,
                "has_conflict": has_conflict,
                "primary_keywords": primary_keywords_str,
                "secondary_keywords": secondary_keywords_str,
                "top2_question": top2_question,
                "top2_score": top2_score,
                "has_coverage": "N/A",
                "candidates_debug": candidates_debug,
            }

    # Reranking
    top_candidates = candidates[:top_k]
    llm_scores: Optional[Dict[str, float]] = None
    is_na = False

    llm_scores = rerank_candidates(query, top_candidates, client, model)

    # 一致性抑制 (Consistency Damping)：當向量檢索的第一名被 Reranker 翻轉時，進行置信度調整。
    # 用以避免 Reranker 在檢索分數差距極大時，因為細微的字面理解偏差而做出錯誤的翻轉。
    # 我們引入「檢索置信度引力」(Retrieval Confidence Gravity)：若向量檢索的第一名與第二名差距顯著（例如 >= 3.0 分），
    # 代表檢索結果相當確定，此時將對 Reranker 的翻轉施加懲罰，限制其隨意翻轉的幅度；若差距巨大（如 >= 10.0），則極難被翻轉。
    if llm_scores and len(top_candidates) >= 2:
        ret_top1_id = top_candidates[0]["record"].record_id
        llm_top1_id = max(llm_scores, key=llm_scores.get)

        if ret_top1_id != llm_top1_id and ret_top1_id in llm_scores:
            llm_gap = llm_scores[llm_top1_id] - llm_scores[ret_top1_id]

            if llm_gap > 0:
                ret_top1_score = top_candidates[0]["total_score"]
                ret_selected_score = next((cand["total_score"] for cand in top_candidates if cand["record"].record_id == llm_top1_id), 0.0)
                ret_diff = ret_top1_score - ret_selected_score

                if ret_diff >= 3.0:
                    # 依據 LLM 評估差距與檢索差距進行 Damping 計算
                    damping = 2.0 + min(llm_gap * 0.6, 12.0) + (ret_diff - 3.0) * 0.8
                elif ret_diff < 1.5:
                    damping = 0.0
                else:
                    damping = min(llm_gap * 0.3, 3.0)

                if damping > 0:
                    llm_scores[llm_top1_id] -= damping
                logging.info(
                    "[Consistency Damping] Reranker flipped retrieval Top-1: "
                    "ret_top1=%s, llm_top1=%s, llm_gap=%.1f, ret_diff=%.2f, damping=%.2f",
                    ret_top1_id, llm_top1_id, llm_gap, ret_diff, damping,
                )

    for item in candidates:
        record_id = item["record"].record_id
        if llm_scores and record_id in llm_scores:
            item["final_score"] = 0.5 * item["total_score"] + 0.5 * llm_scores[record_id]
        else:
            item["final_score"] = item["total_score"]

        # 如果有專利術語衝突，則大幅調低其綜合評分
        if check_conflict(query, item["record"].question):
            item["final_score"] = min(50.0, item["final_score"] - 20.0)

        # 同步更新除錯輸出中的綜合評估數據
        for c_db in candidates_debug:
            if c_db["record_id"] == record_id:
                if llm_scores and record_id in llm_scores:
                    c_db["llm_score"] = round(llm_scores[record_id], 2)
                c_db["final_score"] = round(item["final_score"], 2)

    candidates.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)
    best = candidates[0]

    # 覆蓋度判定優化 (Coverage Bypass)：
    # 若最佳候選條目的向量檢索分數很高 (>= 85.0)，或 >= 80.0 且與次佳候選的差距足夠，且無術語衝突，
    # 則直接將 has_coverage 設為 True，避免呼叫 LLM 進行 check_coverage()，以提升效能並減少 API 開銷。
    best_has_conflict = check_conflict(query, best["record"].question)
    if best["vector_score"] >= 88.0 or (best["vector_score"] >= 88.0 and score_gap_ok and not best_has_conflict):
        has_coverage_val = True
        logging.info(
            "[Coverage Bypass] High confidence vector score (%.2f) and gap ok. Bypassing check_coverage.",
            best["vector_score"]
        )
    else:
        has_coverage_val = check_coverage(query, best["record"], client, model)

    if not has_coverage_val:
        logging.info("[Coverage] Top-1 cannot fully answer. Falling back to Vector Search.")

        # 覆蓋度救援 (Coverage Rescue)：當經過 Reranker 挑選出的最佳條目未通過覆蓋度檢查時，
        # 我們回頭檢查原始向量檢索的第一名。若原始檢索第一名能通過覆蓋度檢查，則救援並採用該條目作為答覆。
        # 這是因為向量檢索在字元語意相似度上更具穩定性，可以避免 Reranker 因為推理偏見而過濾掉正確條目，進而避免誤觸發 RAG。
        retrieval_top1 = max(candidates, key=lambda x: x["total_score"])
        if retrieval_top1["record"].record_id != best["record"].record_id:
            rescue_coverage = check_coverage(query, retrieval_top1["record"], client, model)
            if rescue_coverage:
                logging.info(
                    "[Coverage Rescue] Retrieval Top-1 (ID=%s) passes coverage. Using as standard answer.",
                    retrieval_top1["record"].record_id,
                )
                best = retrieval_top1
                has_coverage_val = True
            else:
                is_na = True
        else:
            is_na = True

    if is_na:
        logging.info("[Coverage/Rerank] No single FAQ can fully answer. Falling back to Vector Search.")
        # 第一階段降級：進行向量庫檢索與 RAG 合成回答
        if VECTOR_SEARCHER:
            top_vec_res = VECTOR_SEARCHER.search_top_k(query, top_k=5)
            valid_res = [res for res in top_vec_res if res[1] >= 0.45]

            if valid_res:
                best_rec, best_score = valid_res[0]
                logging.info("Vector Search Top-1 Score: %.4f, Question: %s", best_score, best_rec.question)

                logging.info("Vector Search hit %d relevant items. Synthesizing answer with LLM...", len(valid_res))
                synthesized_answer, used_ids = synthesize_answer_from_vector(query, valid_res, client, model)
                debug_rows = build_debug_rows(
                    query,
                    [{"record": r, "total_score": s*100, "final_score": s*100} for r, s in valid_res],
                    best_rec.record_id,
                    None,
                    primary_keywords,
                    secondary_keywords
                )
                append_debug_rows(debug_rows, DEBUG_CSV_PATH)
                logging.info("[RAG] Vector Search + LLM Synthesis used. Best score: %.2f%%, Reference IDs: %s", best_score * 100, used_ids)
                return {
                    "answer": synthesized_answer,
                    "full_answer": synthesized_answer,
                    "category": "智慧搜尋結果",
                    "question": query,
                    "score": round(best_score * 100, 2),
                    "low_confidence": False,
                    "reference_ids": used_ids,
                    "response_type": "rag",
                    "retrieval_score": round(best_score * 100, 2),
                    "retrieval_rank": 1,
                    "llm_score": "N/A",
                    "llm_rank": "N/A",
                    "final_score": round(best_score * 100, 2),
                    "final_rank": 1,
                    # 除錯元數據
                    "exact_match": exact_match,
                    "fuzzy_match": fuzzy_match,
                    "semantic_match": semantic_match,
                    "score_gap_ok": score_gap_ok,
                    "has_conflict": has_conflict,
                    "primary_keywords": primary_keywords_str,
                    "secondary_keywords": secondary_keywords_str,
                    "top2_question": top2_question,
                    "top2_score": top2_score,
                    "has_coverage": has_coverage_val,
                    "candidates_debug": candidates_debug,
                }

        # 第二階段降級：降級至使用通用專利知識回答
        logging.info("Vector Search failed or score too low. Falling back to General Knowledge.")
        gen_answer = answer_from_general_knowledge(query, client, model)
        logging.info("[General Knowledge] LLM used to answer from general knowledge.")
        return {
            "answer": gen_answer,
            "full_answer": gen_answer,
            "category": "通用專利知識",
            "question": query,
            "score": 0.0,
            "low_confidence": True,
            "response_type": "general_knowledge",
            "retrieval_score": 0.0,
            "retrieval_rank": "N/A",
            "llm_score": "N/A",
            "llm_rank": "N/A",
            "final_score": 0.0,
            "final_rank": "N/A",
            # 除錯元數據
            "exact_match": exact_match,
            "fuzzy_match": fuzzy_match,
            "semantic_match": semantic_match,
            "score_gap_ok": score_gap_ok,
            "has_conflict": has_conflict,
            "primary_keywords": primary_keywords_str,
            "secondary_keywords": secondary_keywords_str,
            "top2_question": top2_question,
            "top2_score": top2_score,
            "has_coverage": has_coverage_val,
            "candidates_debug": candidates_debug,
        }

    low_confidence = best["final_score"] < low_confidence_threshold

    debug_rows = build_debug_rows(
        query,
        candidates,
        best["record"].record_id,
        llm_scores,
        primary_keywords,
        secondary_keywords,
    )
    append_debug_rows(debug_rows, DEBUG_CSV_PATH)

    # 搜尋最佳條目在原始檢索結果中的相似度與排名
    retrieval_score = 0.0
    retrieval_rank = "N/A"
    for r_idx, cand in enumerate(top_candidates):
        if cand["record"].record_id == best["record"].record_id:
            retrieval_score = cand["total_score"]
            retrieval_rank = r_idx + 1
            break
    if retrieval_rank == "N/A" and best in candidates:
        retrieval_score = best["total_score"]
        retrieval_rank = candidates.index(best) + 1

    # 計算並記錄 LLM Rerank 之後的排名與分數
    llm_score = "N/A"
    llm_rank = "N/A"
    if llm_scores:
        rec_id = best["record"].record_id
        if rec_id in llm_scores:
            llm_score = llm_scores[rec_id]
            
            # 根據 LLM 分數對候選條目進行排序，以計算最佳條目在 Reranker 中的排名
            llm_sorted = sorted(
                [cand for cand in top_candidates if cand["record"].record_id in llm_scores],
                key=lambda x: llm_scores.get(x["record"].record_id, 0.0),
                reverse=True
            )
            for l_idx, cand in enumerate(llm_sorted):
                if cand["record"].record_id == rec_id:
                    llm_rank = l_idx + 1
                    break

    logging.info("[Standard] Selected final answer from FAQ (response_type: standard): %s (Final Score: %.2f)", best["record"].question, best["final_score"])
    return {
        "answer": best["record"].answer,
        "full_answer": format_answer(best["record"], low_confidence),
        "category": best["record"].category,
        "question": best["record"].question,
        "record_id": best["record"].record_id,
        "score": round(best["final_score"], 2),
        "low_confidence": low_confidence,
        "response_type": "standard",
        "retrieval_score": round(retrieval_score, 2),
        "retrieval_rank": retrieval_rank,
        "llm_score": round(llm_score, 2) if isinstance(llm_score, (int, float)) else llm_score,
        "llm_rank": llm_rank,
        "final_score": round(best["final_score"], 2),
        "final_rank": 1,
        # 除錯元數據
        "exact_match": exact_match,
        "fuzzy_match": fuzzy_match,
        "semantic_match": semantic_match,
        "score_gap_ok": score_gap_ok,
        "has_conflict": has_conflict,
        "primary_keywords": primary_keywords_str,
        "secondary_keywords": secondary_keywords_str,
        "top2_question": top2_question,
        "top2_score": top2_score,
        "has_coverage": has_coverage_val,
        "candidates_debug": candidates_debug,
    }


def verify_line_signature(body: bytes, signature: str, channel_secret: str) -> bool:
    if not body or not signature or not channel_secret:
        return False
    digest = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    computed = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(computed, signature)


def reply_line_message(reply_token: str, text: str, access_token: str) -> None:
    if not reply_token or not access_token:
        return
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "replyToken": reply_token,
        "messages": [
            {
                "type": "text",
                "text": text,
            }
        ],
    }
    try:
        requests.post("https://api.line.me/v2/bot/message/reply", headers=headers, json=payload, timeout=10)
    except Exception:
        logging.exception("LINE reply failed")

# LINE credentials - read from environment if available (empty string fallback)
LINE_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
LINE_SECRET = os.getenv("LINE_CHANNEL_SECRET", "").strip()

# LMStudio settings
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LLM_MODEL = "qwen3.5-4b"
LLM_CLIENT = OpenAI(base_url=LMSTUDIO_BASE_URL)
logging.info("Using LM Studio OpenAI SDK at %s with model %s", LMSTUDIO_BASE_URL, LLM_MODEL)

FAQ_RECORDS = load_faq("db.csv")
VECTOR_SEARCHER = VectorSearcher(FAQ_RECORDS, LLM_CLIENT)
BM25_INDEX = BM25Index(FAQ_RECORDS) if FAQ_RECORDS else None


@app.route("/", methods=["GET"])
def index() -> str:
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(silent=True) or {}
    message = payload.get("message", "")
    result = answer_query(message, FAQ_RECORDS, LLM_CLIENT, LLM_MODEL)
    return jsonify(result)


@app.route("/api/faq/<record_id>", methods=["GET"])
def api_get_faq_detail(record_id):
    record = next((r for r in FAQ_RECORDS if r.record_id == record_id), None)
    if not record:
        return jsonify({"error": "Record not found"}), 404
    return jsonify({
        "record_id": record.record_id,
        "category": record.category,
        "question": record.question,
        "answer": record.answer,
        "updated": record.updated
    })


@app.route("/api/faq/random", methods=["GET"])
def api_faq_random():
    import random
    try:
        limit = int(request.args.get("limit", 4))
    except ValueError:
        limit = 4
        
    # 隨機挑選長度在 8 到 35 字之間的常見問題，以確保展示在介面上的效果最佳
    valid_records = [
        r for r in FAQ_RECORDS 
        if r.question and 8 <= len(r.question) <= 35
    ]
    if not valid_records:
        valid_records = FAQ_RECORDS
        
    sampled = random.sample(valid_records, min(len(valid_records), limit))
    return jsonify([{
        "record_id": r.record_id,
        "category": r.category,
        "question": r.question
    } for r in sampled])


@app.route("/api/db/categories", methods=["GET"])
def api_db_categories():
    categories = sorted(list(set(r.category for r in FAQ_RECORDS if r.category)))
    return jsonify(categories)


@app.route("/api/db/search", methods=["GET"])
def api_db_search():
    query = request.args.get("query", "").strip().lower()
    categories_raw = request.args.get("categories", "").strip()
    field = request.args.get("field", "all").strip().lower()
    
    try:
        page = int(request.args.get("page", 1))
        if page < 1:
            page = 1
    except ValueError:
        page = 1
        
    try:
        page_size = int(request.args.get("pageSize", 20))
        if page_size < 1:
            page_size = 20
    except ValueError:
        page_size = 20
    
    selected_categories = [c.strip() for c in categories_raw.split(",") if c.strip()]
    
    filtered = FAQ_RECORDS
    if selected_categories:
        filtered = [r for r in filtered if r.category in selected_categories]
        
    if query:
        if field == "question":
            filtered = [r for r in filtered if query in r.question.lower()]
        elif field == "answer":
            filtered = [r for r in filtered if query in r.answer.lower()]
        else:
            filtered = [
                r for r in filtered
                if query in r.question.lower() or query in r.answer.lower()
            ]
        
    total_matches = len(filtered)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    results = filtered[start_idx:end_idx]
    
    return jsonify({
        "total": total_matches,
        "page": page,
        "pageSize": page_size,
        "results": [{
            "record_id": r.record_id,
            "category": r.category,
            "question": r.question,
            "updated": r.updated
        } for r in results]
    })



@app.route("/callback", methods=["POST"])
def webhook():
    body = request.get_data()
    signature = request.headers.get("x-line-signature", "")
    if not verify_line_signature(body, signature, LINE_SECRET):
        abort(400)

    payload = request.get_json(silent=True) or {}
    events = payload.get("events", [])

    for event in events:
        if event.get("type") != "message":
            continue
        message = event.get("message", {})
        if message.get("type") != "text":
            continue

        reply_token = event.get("replyToken")
        user_text = message.get("text", "")
        result = answer_query(user_text, FAQ_RECORDS, LLM_CLIENT, LLM_MODEL)
        reply_line_message(reply_token, result["full_answer"], LINE_ACCESS_TOKEN)

    return "OK"


if __name__ == "__main__":
    import sys
    # 設定/配置 root logger 以便將日誌正常輸出到 stdout，且避免重複輸出與強制 flush
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.root.addHandler(stdout_handler)
    logging.root.setLevel(logging.INFO)

    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
