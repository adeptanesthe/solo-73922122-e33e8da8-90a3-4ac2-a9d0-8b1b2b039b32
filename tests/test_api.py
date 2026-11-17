"""API 端到端测试：会话创建、校验、判断/参数版本、异步求解与迟到结果作废。"""

import io
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app

client = TestClient(app)


def png_bytes(color=(200, 30, 30)):
    img = Image.new("RGB", (64, 48), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_session(n_imgs=3, dets=None):
    if dets is None:
        dets = [
            {"id": "a", "frame": 0, "rect": [1, 1, 8, 8], "vector": [1, 0]},
            {"id": "b", "frame": 1, "rect": [11, 1, 8, 8], "vector": [1, 0]},
        ]
    files = [("images", (f"f{i}.png", png_bytes(), "image/png")) for i in range(n_imgs)]
    files.append(("detections", ("det.json", json.dumps({"detections": dets}), "application/json")))
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 200, r.text
    return r.json()


def wait_job(job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise TimeoutError("求解任务超时未完成")


# ---------------------------------------------------------------- 会话与校验

def test_create_session_ok():
    p = make_session()
    assert p["n_frames"] == 3 and len(p["detections"]) == 2
    assert p["version"] == 1
    img = client.get(f"/api/sessions/{p['session_id']}/frames/0")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"


def test_too_many_images_rejected():
    files = [("images", (f"f{i}.png", png_bytes(), "image/png")) for i in range(11)]
    files.append(("detections", ("d.json", json.dumps([]), "application/json")))
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "10" in r.json()["detail"]


def test_too_many_detections_rejected():
    dets = [{"id": f"b{i}", "frame": 0, "rect": [0, 0, 4, 4], "vector": [1]} for i in range(31)]
    files = [("images", ("f0.png", png_bytes(), "image/png")),
             ("detections", ("d.json", json.dumps(dets), "application/json"))]
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "30" in r.json()["detail"]


@pytest.mark.parametrize("det,needle", [
    ({"id": "a", "frame": 0, "rect": [0, 0, 4, 4], "vector": [0, 0]}, "非零"),
    ({"id": "a", "frame": 5, "rect": [0, 0, 4, 4], "vector": [1]}, "帧号"),
    ({"id": "a", "frame": 0, "rect": [0, 0, -4, 4], "vector": [1]}, "宽"),
    ({"id": "a", "frame": 0, "rect": [0, 0, 4, 4], "vector": [1, float("nan")]}, "非法"),
])
def test_bad_detection_rejected(det, needle):
    files = [("images", ("f0.png", png_bytes(), "image/png")),
             ("detections", ("d.json", json.dumps([det]), "application/json"))]
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and needle in r.json()["detail"]


def test_vector_dim_mismatch_rejected():
    dets = [
        {"id": "a", "frame": 0, "rect": [0, 0, 4, 4], "vector": [1, 0]},
        {"id": "b", "frame": 0, "rect": [8, 0, 4, 4], "vector": [1, 0, 0]},
    ]
    files = [("images", ("f0.png", png_bytes(), "image/png")),
             ("detections", ("d.json", json.dumps(dets), "application/json"))]
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "维度" in r.json()["detail"]


# ---------------------------------------------------------------- 示例与求解

def test_example_solves_to_two_animals():
    p = client.post("/api/example").json()
    assert p["n_frames"] == 6 and len(p["detections"]) == 11
    job = client.post(f"/api/sessions/{p['session_id']}/solve").json()
    done = wait_job(job["job_id"])
    assert done["status"] == "done"
    res = done["result"]
    assert res["feasible"] and res["animal_count"] == 2 and res["n_trajectories"] == 2
    assert res["total_cost"] == pytest.approx(res["assoc_cost"] + res["penalty_cost"])


def test_conflicting_judgments_report_infeasible_without_count():
    p = client.post("/api/example").json()
    sid = p["session_id"]
    # A0 与 B0 同处第 0 帧，标记同只 → 无解
    r = client.put(f"/api/sessions/{sid}/judgments",
                   json={"judgments": [{"a": "A0", "b": "B0", "type": "same"}]})
    assert r.status_code == 200
    job = client.post(f"/api/sessions/{sid}/solve").json()
    done = wait_job(job["job_id"])
    res = done["result"]
    assert res["feasible"] is False
    assert "animal_count" not in res
    assert "无解" in res["message"]


def test_same_judgment_changes_result():
    p = client.post("/api/example").json()
    sid = p["session_id"]
    # 把 B0 与 A1 判为同只（它们帧差 1），全局解必须让二者同轨
    client.put(f"/api/sessions/{sid}/judgments",
               json={"judgments": [{"a": "B0", "b": "A1", "type": "same"}]})
    done = wait_job(client.post(f"/api/sessions/{sid}/solve").json()["job_id"])
    res = done["result"]
    assert res["feasible"]
    for t in res["trajectories"]:
        ids = {b["id"] for b in t["boxes"]}
        assert not ({"B0", "A1"} <= ids) or True  # 占位，下面统一检查
    same_traj = [t for t in res["trajectories"]
                 if {"B0", "A1"} <= {b["id"] for b in t["boxes"]}]
    assert len(same_traj) == 1


# ---------------------------------------------------------------- 版本与迟到结果

def test_version_bumps_and_stale_job_detectable():
    p = make_session()
    sid = p["session_id"]
    assert p["version"] == 1

    r = client.put(f"/api/sessions/{sid}/params",
                   json={"w_dist": 2.0, "w_app": 50.0, "penalty": 100.0})
    assert r.status_code == 200 and r.json()["version"] == 2

    job = client.post(f"/api/sessions/{sid}/solve").json()
    assert job["version"] == 2  # 任务记录启动时版本

    # 求解期间继续修改 → 版本再 +1，旧任务结果即为“迟到结果”
    r = client.put(f"/api/sessions/{sid}/judgments",
                   json={"judgments": [{"a": "a", "b": "b", "type": "different"}]})
    assert r.json()["version"] == 3

    done = wait_job(job["job_id"])
    assert done["version"] == 2  # 结果仍携带旧版本号
    current = client.get(f"/api/sessions/{sid}").json()
    assert current["version"] == 3
    assert done["version"] != current["version"]  # 前端据此丢弃，不覆盖新输入


def test_params_validation():
    p = make_session()
    r = client.put(f"/api/sessions/{p['session_id']}/params",
                   json={"w_dist": -1, "w_app": 1, "penalty": 1})
    assert r.status_code == 422


def test_unknown_session_404():
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.post("/api/sessions/nope/solve").status_code == 404


def test_healthz():
    assert client.get("/healthz").json() == {"ok": True}


# ---------------------------------------------------------------- 点选分割

def _wait_job(job_id):
    return wait_job(job_id)


def test_segment_endpoint_success():
    p = client.post("/api/example").json()
    sid = p["session_id"]
    dets_before = client.get(f"/api/sessions/{sid}/detections.json").json()
    version_before = client.get(f"/api/sessions/{sid}").json()["version"]

    # 示例第 0 帧：A（红圆 r=14，中心 (40,60)），草地绿背景；框 28×28
    body = {
        "detection_id": "A0",
        "foreground": [{"x": 36, "y": 60}, {"x": 44, "y": 60}],
        "background": [{"x": 29, "y": 49}],
    }
    r = client.post(f"/api/sessions/{sid}/segment", json=body)
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["kind"] == "segment" and job["max_crop"] == 256

    done = _wait_job(job["job_id"])
    assert done["status"] == "done"
    res = done["result"]
    assert res["ok"] is True
    assert res["area_px"] > 0
    assert res["crop"]["width"] == 28 and res["crop"]["height"] == 28
    # RLE 行程总长等于裁剪像素数，且以 off/on 交替表示
    assert sum(res["mask_rle"]) == 28 * 28

    # 分割只读数据：检测 JSON 与会话版本不变
    dets_after = client.get(f"/api/sessions/{sid}/detections.json").json()
    assert dets_after == dets_before
    assert client.get(f"/api/sessions/{sid}").json()["version"] == version_before


def test_segment_solid_color_reports_unreliable_no_area():
    # 纯色图：颜色无法区分前后景，返回 ok=False 且不带面积
    files = [("images", ("f0.png", png_bytes(), "image/png")),
             ("detections", ("d.json", json.dumps({
                 "detections": [{"id": "a", "frame": 0, "rect": [2, 2, 20, 20],
                                 "vector": [1, 0]}]}), "application/json"))]
    p = client.post("/api/sessions", files=files).json()
    r = client.post(f"/api/sessions/{p['session_id']}/segment", json={
        "detection_id": "a",
        "foreground": [{"x": 5, "y": 5}, {"x": 10, "y": 10}],
        "background": [{"x": 18, "y": 18}],
    })
    done = _wait_job(r.json()["job_id"])
    res = done["result"]
    assert res["ok"] is False and "area_px" not in res
    assert "纯色" in res["message"]


def test_segment_oversized_box_rejected_via_job():
    # 大检测框：裁剪超过 256×256，任务结果给出提示而不是面积
    big = Image.new("RGB", (300, 100), (10, 20, 10))
    buf = io.BytesIO()
    big.save(buf, format="PNG")
    files = [("images", ("big.png", buf.getvalue(), "image/png")),
             ("detections", ("d.json", json.dumps({
                 "detections": [{"id": "big", "frame": 0, "rect": [0, 0, 260, 90],
                                 "vector": [1]}]}), "application/json"))]
    p = client.post("/api/sessions", files=files).json()
    r = client.post(f"/api/sessions/{p['session_id']}/segment", json={
        "detection_id": "big",
        "foreground": [{"x": 10, "y": 10}, {"x": 20, "y": 10}],
        "background": [{"x": 50, "y": 50}],
    })
    res = _wait_job(r.json()["job_id"])["result"]
    assert res["ok"] is False and "area_px" not in res
    assert "256" in res["message"]


def test_segment_broken_image_reports_decode_failure():
    files = [("images", ("broken.png", b"garbage-not-image", "image/png")),
             ("detections", ("d.json", json.dumps({
                 "detections": [{"id": "a", "frame": 0, "rect": [0, 0, 20, 20],
                                 "vector": [1]}]}), "application/json"))]
    p = client.post("/api/sessions", files=files).json()
    r = client.post(f"/api/sessions/{p['session_id']}/segment", json={
        "detection_id": "a",
        "foreground": [{"x": 5, "y": 5}, {"x": 10, "y": 10}],
        "background": [{"x": 15, "y": 15}],
    })
    res = _wait_job(r.json()["job_id"])["result"]
    assert res["ok"] is False and "解码" in res["message"]


def test_segment_validation_errors():
    p = make_session()
    sid = p["session_id"]
    # 未知检测框
    r = client.post(f"/api/sessions/{sid}/segment", json={
        "detection_id": "nope",
        "foreground": [{"x": 2, "y": 2}, {"x": 3, "y": 3}],
        "background": [{"x": 6, "y": 6}],
    })
    assert r.status_code == 400
    # 标点落在框外（框 a 为 [1,1,8,8]，x/y 范围 1~9）
    r = client.post(f"/api/sessions/{sid}/segment", json={
        "detection_id": "a",
        "foreground": [{"x": 2, "y": 2}, {"x": 3, "y": 3}],
        "background": [{"x": 50, "y": 50}],
    })
    assert r.status_code == 400 and "外的标点" in r.json()["detail"]
    # 前景点不足 2 个 → pydantic 422
    r = client.post(f"/api/sessions/{sid}/segment", json={
        "detection_id": "a",
        "foreground": [{"x": 2, "y": 2}],
        "background": [{"x": 6, "y": 6}],
    })
    assert r.status_code == 422
    # 非法坐标
    r = client.post(f"/api/sessions/{sid}/segment", json={
        "detection_id": "a",
        "foreground": [{"x": 2, "y": 2}, {"x": None, "y": 3}],
        "background": [{"x": 6, "y": 6}],
    })
    assert r.status_code == 422


def test_segment_jobs_are_independent_and_pollable():
    # 两次提交互不覆盖：各自任务 id 独立，结果分别可取
    p = client.post("/api/example").json()
    sid = p["session_id"]
    body = {
        "detection_id": "A0",
        "foreground": [{"x": 36, "y": 60}, {"x": 44, "y": 60}],
        "background": [{"x": 29, "y": 49}],
    }
    j1 = client.post(f"/api/sessions/{sid}/segment", json=body).json()["job_id"]
    j2 = client.post(f"/api/sessions/{sid}/segment", json=body).json()["job_id"]
    assert j1 != j2
    r1, r2 = _wait_job(j1)["result"], _wait_job(j2)["result"]
    assert r1["ok"] and r2["ok"]
    assert r1["area_px"] == r2["area_px"]
    # 未知任务 404
    assert client.get("/api/jobs/nope").status_code == 404
