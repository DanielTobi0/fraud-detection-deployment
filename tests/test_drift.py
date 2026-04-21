from src.drift import DriftMonitor


def test_drift_stays_ok_for_small_shift() -> None:
    monitor = DriftMonitor(reference_mean=0.2, reference_std=0.1, window_size=40)
    for _ in range(40):
        monitor.add_score(0.21)

    result = monitor.evaluate()
    assert result.status == "ok"


def test_drift_detected_for_large_shift() -> None:
    monitor = DriftMonitor(reference_mean=0.2, reference_std=0.05, window_size=40)
    for _ in range(40):
        monitor.add_score(0.5)

    result = monitor.evaluate()
    assert result.status == "detected"
