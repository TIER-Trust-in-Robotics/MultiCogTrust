import json
import re
import urllib.error
import urllib.request

import pytest

from src.inspector_dashboard import InspectorDashboard


def test_inspector_bounds_history_serves_state_and_controls_only_its_session():
    dashboard = InspectorDashboard(port=0)
    stopped = []
    dashboard.set_stop_callback(lambda: stopped.append(True))
    try:
        for index in range(25):
            dashboard.record(
                {
                    "event": "prediction",
                    "text": f"utterance {index}",
                    "start_sec": index,
                    "end_sec": index + 1,
                    "emotion": None,
                    "status": "empty_transcript",
                }
            )
        dashboard.vision({"status": "no_face"}, b"preview-placeholder")
        response = urllib.request.urlopen(dashboard.url + "/state")
        assert response.headers["Cache-Control"] == "no-store"
        state = json.load(response)
        assert len(state["history"]) == 20
        assert state["history"][0]["text"] == "utterance 24"
        assert state["vision"]["status"] == "no_face"
        html = urllib.request.urlopen(dashboard.url).read().decode()
        token = re.search("'X-Session':'([^']+)'", html)[1]
        with pytest.raises(urllib.error.HTTPError) as failure:
            urllib.request.urlopen(
                urllib.request.Request(dashboard.url + "/stop", method="POST")
            )
        assert failure.value.code == 403 and not stopped
        request = urllib.request.Request(
            dashboard.url + "/stop", method="POST", headers={"X-Session": token}
        )
        assert urllib.request.urlopen(request).status == 200
        assert stopped == [True] and dashboard.stop_requested.is_set()
    finally:
        dashboard.close()
