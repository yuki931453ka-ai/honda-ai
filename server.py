import os
import json
import uuid
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple, Union

from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
import anthropic

load_dotenv()

_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("ANTHROPIC_API_KEY が設定されていません")

app = Flask(__name__, static_folder=".", static_url_path="")
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
limiter = Limiter(get_remote_address, app=app, default_limits=["200 per day"])

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

FEEDBACK_FILE = Path("/tmp/feedback.jsonl") if os.environ.get("VERCEL") else DATA_DIR / "feedback.jsonl"
TRAINING_DATA_FILE = DATA_DIR / "honda_training_data.md"

_feedback_lock = threading.Lock()
_prompt_lock = threading.Lock()

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
FEEDBACK_READ_TOKEN = os.environ.get("FEEDBACK_READ_TOKEN", "")

MAX_MESSAGES = 50
MAX_MSG_CONTENT_LEN = 8_000
MAX_FIELD_LEN = 10_000
ALLOWED_ROLES = {"user", "assistant"}

client = anthropic.Anthropic(api_key=api_key)

SYSTEM_PROMPT = """あなたは「本田 覚（ほんだ さとる）」です。
Nu Skin Japan のチームエリート（最高ランク）を複数回達成した実績を持ち、
(株)PLEASURE に所属するトップリーダーです。

## あなたの人物像
- 2010年頃からNu Skinでの活動を開始し、15年以上の経験がある
- Team Elite を2018年（8年目）、2019年（9年目）、2021年、2023年、2024年に達成
- (株)VEEDLE JAPAN（高倉慎仁 下）から (株)PLEASURE へ所属変遷
- パートナーの「のとか」さんと共に活動
- 東日本大震災被災地支援（陸前高田・大船渡・気仙沼）に複数回参加
- Force for Good Foundation に10万円以上の寄付を継続
- Instagram: @pleasure0722

## あなたの価値観・信念（発言より）
- 「皆様のご協力があれば、もっと良いまちに変わると信じています」
- 「新しい本に目を輝かせ喜ぶ姿を見て、ニュースキンの活動がたくさんの笑顔と未来をつくるきっかけになっていることを肌で感じた」
- 社会貢献活動を通じた「笑顔と未来の創出」を大切にしている
- 長期にわたり最高ランクを維持する粘り強さと継続力
- 人とのつながり、チームの力を信じる

## Pleasureの理念
1. 目の前の人に喜びを与える
2. 成功者を増やす
3. 成幸家族を増やす

## 会話のスタイル
- 丁寧で分かりやすく、優しくフランクな口調で話す
- 温かく誠実で、相手の話をしっかり聞く姿勢を持つ
- 自分の経験に基づいた具体的なアドバイスをする
- 押しつけがましくなく、相手のペースを尊重する
- ビジネスの話だけでなく、人生や家族の話も大切にする
- 日本語で自然に会話する

## 問いかけのスタンス（最重要）
- 一方的に語るだけでなく、適切な問いかけを通じて相手自身に気づきを促すことを最優先する
- 応答の最後には、相手の思いや優先順位を整理する手助けとなる問いかけを置く（必要な場合）
- 「棚卸し」の姿勢：相手が自分の本当の気持ちや大切にしていることを言語化できるようサポートする
- 問いかけは詰問にならないよう、温かく寄り添うトーンで行う
- 相手の答えを急かさず、考える余白を大切にする

## 重要な注意
- あなたは本田覚本人として振る舞ってください
- 「AIです」「言語モデルです」などとは言わないでください
- 知らないことを聞かれたら「それについてはちょっと分からないな」と自然に答えてください
- 過度に丁寧すぎず、親しみやすい口調で話してください
"""


def load_training_data() -> str:
    if TRAINING_DATA_FILE.exists():
        content = TRAINING_DATA_FILE.read_text(encoding="utf-8")
        return f"\n\n## 講演・発言データ（学習素材）\n{content}"
    return ""


_prompt_cache: dict = {}

def build_system_prompt() -> str:
    mtime = TRAINING_DATA_FILE.stat().st_mtime if TRAINING_DATA_FILE.exists() else 0
    with _prompt_lock:
        if _prompt_cache.get("mtime") != mtime:
            extra = load_training_data()
            _prompt_cache["mtime"] = mtime
            _prompt_cache["prompt"] = SYSTEM_PROMPT + extra
        return _prompt_cache["prompt"]


def build_tone_instruction(strictness: int) -> str:
    if strictness <= 15:
        return (
            "\n\n## トーン指示（最優先で従うこと）\n"
            "- とにかく優しく、絶対に相手を傷つけない言葉選びをする\n"
            "- 丁寧で冗長でも構わないので、相手が安心できるよう包み込むように話す\n"
            "- 否定的な表現は避け、すべてポジティブに言い換える\n"
            "- 相手のどんな意見も一度受け止めてから話す\n"
            "- 問いかけも柔らかく、プレッシャーにならないよう配慮する\n"
        )
    if strictness <= 35:
        return (
            "\n\n## トーン指示（最優先で従うこと）\n"
            "- 優しく穏やかな口調を基本とする\n"
            "- 説明は丁寧に、相手が理解できるよう言葉を尽くす\n"
            "- 厳しいことを伝える必要がある場合も、柔らかい表現で伝える\n"
            "- 相手の気持ちに十分配慮しながら話す\n"
        )
    if strictness <= 65:
        return (
            "\n\n## トーン指示（最優先で従うこと）\n"
            "- 優しさと率直さのバランスを取る\n"
            "- 相手を思いやりつつも、伝えるべきことははっきり伝える\n"
            "- 適度な長さで、要点を押さえた話し方をする\n"
        )
    if strictness <= 85:
        return (
            "\n\n## トーン指示（最優先で従うこと）\n"
            "- ストレートに本質を突く話し方をする\n"
            "- 回りくどい表現は避け、簡潔に要点を伝える\n"
            "- 甘やかさず、相手の成長のために必要な厳しさを持つ\n"
            "- ただし人格否定はせず、行動や考え方に対してフィードバックする\n"
        )
    return (
        "\n\n## トーン指示（最優先で従うこと）\n"
        "- 非常にストレートで簡潔な口調で話す\n"
        "- 無駄な前置きや慰めは省き、核心をズバッと言い切る\n"
        "- 「それは違う」「甘い」など、厳しい言葉も躊躇なく使う\n"
        "- 相手に気づきを与えるためなら厳しさを恐れない\n"
        "- ただし人格攻撃は絶対にしない。あくまで愛のある厳しさ\n"
    )


def parse_strictness(value: Any) -> int:
    if not isinstance(value, (int, float)):
        return 50
    return max(0, min(100, int(value)))


def validate_messages(messages: Any) -> Tuple[bool, str]:
    if not isinstance(messages, list):
        return False, "messages は配列で必須です"
    if len(messages) == 0:
        return False, "messages は1件以上必要です"
    if len(messages) > MAX_MESSAGES:
        return False, f"メッセージ数の上限は {MAX_MESSAGES} 件です"
    if messages[0].get("role") != "user":
        return False, "最初のメッセージは user role である必要があります"
    for msg in messages:
        if not isinstance(msg, dict):
            return False, "不正なメッセージ形式です"
        if msg.get("role") not in ALLOWED_ROLES:
            return False, f"不正なrole: {msg.get('role')}"
        content = msg.get("content", "")
        if not isinstance(content, str) or len(content) == 0 or len(content) > MAX_MSG_CONTENT_LEN:
            return False, f"contentは1文字以上 {MAX_MSG_CONTENT_LEN} 文字以内の文字列で必須です"
    return True, ""


def truncate(value: Any, max_len: int = MAX_FIELD_LEN) -> str:
    return str(value)[:max_len]


BLOCKED_EXTENSIONS = {".py", ".env", ".jsonl", ".md"}
BLOCKED_PREFIXES = ["/data/", "/."]

@app.before_request
def block_sensitive_files() -> Optional[Tuple[Response, int]]:
    path = request.path.lower()
    for prefix in BLOCKED_PREFIXES:
        if path.startswith(prefix):
            return jsonify({"error": "Not Found"}), 404
    for ext in BLOCKED_EXTENSIONS:
        if path.endswith(ext):
            return jsonify({"error": "Not Found"}), 404
    return None


@app.after_request
def set_security_headers(response: Response) -> Response:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "connect-src 'self'; "
        "img-src 'self' data:; "
        "frame-ancestors 'none';"
    )
    return response


@app.route("/")
def index() -> Response:
    return send_from_directory(".", "index.html")


@app.route("/api/chat", methods=["POST"])
@limiter.limit("20 per minute")
def chat() -> Union[Tuple[Response, int], Response]:
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "JSON body が必要です"}), 400

    messages = body.get("messages", [])
    if not messages:
        return jsonify({"error": "messages is required"}), 400

    valid, err_msg = validate_messages(messages)
    if not valid:
        return jsonify({"error": err_msg}), 400

    strictness = parse_strictness(body.get("strictness", 50))

    system = build_system_prompt() + build_tone_instruction(strictness)

    # advisor-strategy: skipped (advisor_20260301 not available on this API key)
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=system,
            messages=messages,
        )
    except anthropic.APIError as e:
        logger.error("Anthropic API error: %s", e)
        return jsonify({"error": "AIサービスで一時的なエラーが発生しました"}), 502
    except Exception as e:
        logger.exception("Unexpected error in /api/chat")
        return jsonify({"error": "サーバー内部エラーが発生しました"}), 500

    assistant_text = ""
    for block in response.content:
        if block.type == "text":
            assistant_text += block.text

    return jsonify({
        "id": response.id,
        "text": assistant_text,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    })


@app.route("/api/feedback", methods=["POST"])
@limiter.limit("30 per minute")
def feedback() -> Union[Tuple[Response, int], Response]:
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "JSON body が必要です"}), 400

    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": truncate(body.get("session_id", ""), 64),
        "message_id": truncate(body.get("message_id", ""), 64),
        "user_message": truncate(body.get("user_message", "")),
        "ai_response": truncate(body.get("ai_response", "")),
        "rating": truncate(body.get("rating", ""), 32),
        "comment": truncate(body.get("comment", ""), 2_000),
        "reviewer": truncate(body.get("reviewer", "anonymous"), 128),
        "strictness": parse_strictness(body.get("strictness", 50)),
    }
    with _feedback_lock:
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return jsonify({"status": "ok", "id": entry["id"]})


@app.route("/api/feedback", methods=["GET"])
@limiter.limit("30 per minute")
def list_feedback() -> Union[Tuple[Response, int], Response]:
    token = request.headers.get("X-Admin-Token", "")
    if not FEEDBACK_READ_TOKEN or token != FEEDBACK_READ_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401

    if not FEEDBACK_FILE.exists():
        return jsonify([])
    entries = []
    lines = FEEDBACK_FILE.read_text(encoding="utf-8").strip().split("\n")
    for line in lines[-500:]:
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("フィードバックファイルの破損行をスキップ: %r", line[:100])
    return jsonify(entries)


if __name__ == "__main__":
    logger.info("Honda AI Server starting on http://localhost:5177")
    logger.info("Training data: %s", TRAINING_DATA_FILE)
    logger.info("Feedback log:  %s", FEEDBACK_FILE)
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(host=host, port=5177, debug=False)
