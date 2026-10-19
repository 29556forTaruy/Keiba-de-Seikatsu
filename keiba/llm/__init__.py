"""LLM作業者 (Claude / Gemini / OpenAI の API と、各社CLI).

役割 (config.toml の [workers]):
  news    直前情報の収集 (Web検索)
  predict 統計モデルとデータを読んで印・買い目・見解を書く
  review  別の作業者による予想のセカンドオピニオン
"""

from keiba.llm.base import Worker, WorkerError, WorkerResult, make_worker

__all__ = ["Worker", "WorkerError", "WorkerResult", "make_worker"]
