import base64
import csv
import hashlib
import hmac
import json
import logging
import os
import re
import sys
import random
from typing import Dict, List, Optional, Tuple, Union

import requests
from flask import Flask, abort, jsonify, render_template, request
from openai import OpenAI
from dotenv import load_dotenv

# 導入核心檢索與 AI 引擎
import engine

load_dotenv()

app = Flask(__name__)

# 設定/配置 root logger 以便將日誌正常輸出到 stdout
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.root.addHandler(stdout_handler)
logging.root.setLevel(logging.INFO)


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

# 初始化核心引擎中的全域檢索與快取變數
engine.FAQ_RECORDS = engine.load_faq("db.csv")
engine.VECTOR_SEARCHER = engine.VectorSearcher(engine.FAQ_RECORDS, LLM_CLIENT)
engine.BM25_INDEX = engine.BM25Index(engine.FAQ_RECORDS) if engine.FAQ_RECORDS else None


@app.route("/", methods=["GET"])
def index() -> str:
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(silent=True) or {}
    message = payload.get("message", "")
    result = engine.answer_query(message, engine.FAQ_RECORDS, LLM_CLIENT, LLM_MODEL)
    return jsonify(result)


@app.route("/api/faq/<record_id>", methods=["GET"])
def api_get_faq_detail(record_id):
    record = next((r for r in engine.FAQ_RECORDS if r.record_id == record_id), None)
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
    try:
        limit = int(request.args.get("limit", 4))
    except ValueError:
        limit = 4
        
    # 隨機挑選長度在 8 到 35 字之間的常見問題，以確保展示在介面上的效果最佳
    valid_records = [
        r for r in engine.FAQ_RECORDS 
        if r.question and 8 <= len(r.question) <= 35
    ]
    if not valid_records:
        valid_records = engine.FAQ_RECORDS
        
    sampled = random.sample(valid_records, min(len(valid_records), limit))
    return jsonify([{
        "record_id": r.record_id,
        "category": r.category,
        "question": r.question
    } for r in sampled])


@app.route("/api/db/categories", methods=["GET"])
def api_db_categories():
    categories = sorted(list(set(r.category for r in engine.FAQ_RECORDS if r.category)))
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
    
    filtered = engine.FAQ_RECORDS
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
        result = engine.answer_query(user_text, engine.FAQ_RECORDS, LLM_CLIENT, LLM_MODEL)
        reply_line_message(reply_token, result["full_answer"], LINE_ACCESS_TOKEN)

    return "OK"


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
