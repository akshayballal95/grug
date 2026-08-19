"""Registry behaviour: registration, lookup, discovery, and dependency errors."""

from __future__ import annotations

from typing import ClassVar

import pytest

from grug import registry
from grug.base import CompressionResult, CompressorBackend, MissingDependencyError
from grug.registry import (
    BackendNotFoundError,
    backend_info,
    create_backend,
    default_backend_name,
    get_backend_class,
    list_backends,
    register_backend,
    unregister_backend,
)


class _Dummy(CompressorBackend):
    name = "dummy-test-backend"
    description = "test only"

    def compress(self, text: str, rate: float = 0.5, **kwargs) -> CompressionResult:
        return CompressionResult.build(text, text, self.name)


@pytest.fixture
def dummy():
    register_backend(_Dummy)
    yield _Dummy
    unregister_backend(_Dummy.name)


def test_builtin_backends_are_registered():
    names = list_backends()
    assert "rules" in names
    assert "classifier" in names


def test_preferred_backends_are_listed_first():
    """Built-ins in preference order, third-party registrations after them."""
    names = list_backends()
    assert names[:2] == ["rules", "classifier"]


def test_register_and_create(dummy):
    assert "dummy-test-backend" in list_backends()
    instance = create_backend("dummy-test-backend")
    assert isinstance(instance, _Dummy)
    assert instance.compress("hello world").text == "hello world"


def test_unregister_removes_backend(dummy):
    unregister_backend("dummy-test-backend")
    assert "dummy-test-backend" not in list_backends()


def test_register_requires_a_name():
    class Nameless(CompressorBackend):
        def compress(self, text, rate=0.5, **kwargs):  # pragma: no cover - never called
            raise NotImplementedError

    with pytest.raises(ValueError, match="non-empty class attribute 'name'"):
        register_backend(Nameless)


def test_register_requires_the_abc():
    class NotABackend:
        name = "impostor"

    with pytest.raises(TypeError, match="must subclass CompressorBackend"):
        register_backend(NotABackend)  # type: ignore[arg-type]


def test_unknown_backend_lists_alternatives():
    with pytest.raises(BackendNotFoundError) as excinfo:
        get_backend_class("does-not-exist")
    message = str(excinfo.value)
    assert "does-not-exist" in message
    assert "rules" in message


def test_missing_dependency_error_names_the_extra():
    """A backend whose deps are absent must say which extra installs them."""
    classifier = get_backend_class("classifier")
    if classifier.is_available():
        pytest.skip("torch is installed; nothing to report as missing")

    with pytest.raises(MissingDependencyError) as excinfo:
        create_backend("classifier")
    message = str(excinfo.value)
    assert "pip install 'grugify[classifier]'" in message
    assert "torch" in message


def test_backend_info_reports_availability():
    rows = {row["name"]: row for row in backend_info()}
    assert rows["rules"]["available"] is True
    assert rows["rules"]["extra"] is None
    assert rows["classifier"]["extra"] == "classifier"
    assert rows["rules"]["description"]


def test_default_backend_is_installed():
    name = default_backend_name()
    assert get_backend_class(name).is_available()


def test_entry_point_group_is_discovered(monkeypatch):
    """A third-party package advertising grug.backends is picked up."""

    class _Plugin(CompressorBackend):
        name = "plugin-test-backend"

        def compress(self, text, rate=0.5, **kwargs):
            return CompressionResult.build(text, text[:5], self.name)

    class _EntryPoint:
        name = "plugin-test-backend"

        def load(self):
            return _Plugin

    def fake_entry_points(*, group):
        assert group == registry.ENTRY_POINT_GROUP
        return [_EntryPoint()]

    monkeypatch.setattr(registry, "_entry_points_loaded", False)
    monkeypatch.setattr("importlib.metadata.entry_points", fake_entry_points)
    try:
        registry._load_entry_points()
        assert "plugin-test-backend" in list_backends()
        assert create_backend("plugin-test-backend").compress("abcdefgh").text == "abcde"
    finally:
        unregister_backend("plugin-test-backend")


def test_broken_entry_point_warns_but_does_not_crash(monkeypatch):
    class _BrokenEntryPoint:
        name = "broken"

        def load(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(registry, "_entry_points_loaded", False)
    monkeypatch.setattr("importlib.metadata.entry_points", lambda *, group: [_BrokenEntryPoint()])

    with pytest.warns(RuntimeWarning, match="broken"):
        registry._load_entry_points()
    assert "rules" in list_backends()


def test_importing_grug_does_not_import_torch():
    """The whole point of lazy backends: no ML stack at import time."""
    import subprocess
    import sys

    code = (
        "import sys, grug; "
        "grug.list_backends(); "
        "print([m for m in ('torch', 'transformers') if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# -- constructor kwargs through the one-shot helper -------------------------


def _probe_backend():
    """A backend that records what reached its constructor and its compress()."""
    from grug.base import CompressionResult, CompressorBackend

    class _Probe(CompressorBackend):
        name = "kwargs-probe"
        instances: ClassVar[list] = []

        def __init__(self, **kwargs):
            self.ctor_kwargs = kwargs
            self.calls: list[dict] = []
            type(self).instances.append(self)

        def compress(self, text, rate=0.5, **kwargs):
            self.calls.append(kwargs)
            return CompressionResult.build(text, text, self.name)

    return _Probe


def test_construction_kwargs_reach_the_constructor_through_compress():
    import grug
    from grug.registry import register_backend, unregister_backend

    probe = register_backend(_probe_backend())
    try:
        grug.compress("some text here", backend="kwargs-probe", backend_kwargs={"device": "cuda"})
        assert probe.instances[-1].ctor_kwargs == {"device": "cuda"}
        assert probe.instances[-1].calls == [{}]
    finally:
        unregister_backend("kwargs-probe")
        grug._BACKEND_CACHE.clear()


def test_a_construction_only_kwarg_passed_per_call_is_rejected():
    """Silently forwarding 'device' to compress() ran on the wrong device."""
    import grug
    from grug.registry import register_backend, unregister_backend

    register_backend(_probe_backend())
    try:
        with pytest.raises(TypeError, match="backend_kwargs"):
            grug.compress("some text here", backend="kwargs-probe", device="cuda")
    finally:
        unregister_backend("kwargs-probe")
        grug._BACKEND_CACHE.clear()


def test_differing_construction_kwargs_do_not_share_a_cached_backend():
    import grug
    from grug.registry import register_backend, unregister_backend

    probe = register_backend(_probe_backend())
    try:
        grug.compress("text one here", backend="kwargs-probe", backend_kwargs={"device": "cpu"})
        grug.compress("text two here", backend="kwargs-probe", backend_kwargs={"device": "cuda"})
        assert [i.ctor_kwargs["device"] for i in probe.instances] == ["cpu", "cuda"]
    finally:
        unregister_backend("kwargs-probe")
        grug._BACKEND_CACHE.clear()


def test_list_valued_construction_kwargs_are_accepted():
    """force_tokens is a documented list-typed constructor argument."""
    import grug
    from grug.registry import register_backend, unregister_backend

    probe = register_backend(_probe_backend())
    try:
        grug.compress(
            "some text here", backend="kwargs-probe", backend_kwargs={"force_tokens": ["no", "not"]}
        )
        assert probe.instances[-1].ctor_kwargs == {"force_tokens": ["no", "not"]}
    finally:
        unregister_backend("kwargs-probe")
        grug._BACKEND_CACHE.clear()
