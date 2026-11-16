"""`sec-agent index`: chunk the corpus, close the store, then sync the index. All faked."""

from __future__ import annotations

from typer.testing import CliRunner

import sec_agent.cli as cli
from sec_agent.cli import app
from sec_agent.ingest.index import IndexSummary
from sec_agent.ingest.pipeline import FilingRejectedError

runner = CliRunner()


class FakeStore:
    open_now = False

    def __init__(self, path) -> None:
        self.path = path

    def __enter__(self):
        FakeStore.open_now = True
        return self

    def __exit__(self, *exc) -> None:
        FakeStore.open_now = False

    def chunks(self):
        return ["chunk-1", "chunk-2", "chunk-3"]


class FakeIndex:
    synced: list | None = None
    closed = False
    store_open_during_sync: bool | None = None

    def __init__(self, path, embedder) -> None:
        pass

    def sync(self, chunks):
        FakeIndex.synced = list(chunks)
        FakeIndex.store_open_during_sync = FakeStore.open_now
        return IndexSummary(added=1, updated=0, removed=0, unchanged=2)

    def close(self) -> None:
        FakeIndex.closed = True


def _fake(monkeypatch, settings, chunk_corpus) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli.Corpus, "load", staticmethod(lambda path: "corpus"))
    monkeypatch.setattr(cli, "FactStore", FakeStore)
    monkeypatch.setattr(cli, "chunk_corpus", chunk_corpus)
    monkeypatch.setattr(cli, "load_token_counter", lambda: len)
    monkeypatch.setattr(cli, "ChunkIndex", FakeIndex)
    monkeypatch.setattr(cli, "BgeEmbedder", lambda: "embedder")
    FakeIndex.synced, FakeIndex.closed = None, False


def test_index_chunks_then_syncs_with_the_store_closed(monkeypatch, settings):
    _fake(monkeypatch, settings, lambda corpus, parsed, store, count: {"a": 2, "b": 1})
    result = runner.invoke(app, ["index"])
    assert result.exit_code == 0, result.output
    assert "Chunked 2 filing(s) into 3 chunks." in result.output
    assert "1 added, 0 updated, 0 removed, 2 unchanged" in result.output
    assert FakeIndex.synced == ["chunk-1", "chunk-2", "chunk-3"]
    assert FakeIndex.store_open_during_sync is False and FakeIndex.closed


def test_index_stops_on_a_rejected_filing(monkeypatch, settings):
    def rejected(corpus, parsed, store, count):
        raise FilingRejectedError("0000-1 has no current facts. Run `sec-agent ingest`.")

    _fake(monkeypatch, settings, rejected)
    result = runner.invoke(app, ["index"])
    assert result.exit_code == 1
    assert "Not indexed" in result.output and "sec-agent ingest" in result.output
    assert FakeIndex.synced is None
