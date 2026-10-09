"""SVJ §22 visual bindings (v0.99.2): where each bound part goes and how it moves."""

from __future__ import annotations

import pytest

from svjsim.server import find_visual_bindings


def _bind(doc, *keys, **visual):
    target = doc
    for key in keys:
        target = target[key]
    target["visual"] = {"mesh_ref": "car", **visual}


@pytest.fixture
def bound(document):
    document["assets"] = {"meshes": [{"id": "car", "uri": "car.glb"}]}
    _bind(document, "chassis", node="SVJ::body::chassis")
    _bind(document, "suspension", "FL", "topology", "upright", node="SVJ::suspension::upright_fl")
    _bind(document, "suspension", "FL", "wheel", node="SVJ::wheel::wheel_fl")
    _bind(document, "suspension", "FL", "brake", "disc", node="SVJ::brake::disc_fl")
    _bind(document, "suspension", "FL", "brake", "caliper", node="SVJ::brake::caliper_fl")
    _bind(document, "suspension", "FL", "spring", node="SVJ::suspension::spring_fl")
    _bind(document, "suspension", "FL", "topology", "links", 0,
          node="SVJ::suspension::upper_wishbone_fl", placement="link_between_points", mesh_axis="+y")
    _bind(document, "suspension", "FL", "topology", "links", 1,
          node="SVJ::suspension::lower_wishbone_fl")
    _bind(document, "steering", node="SVJ::steering::rack")
    _bind(document, "powertrain", "engine", node="SVJ::powertrain::engine")
    return document


def _by_node(bindings):
    return {b["node"]: b for b in bindings}


def test_each_part_moves_the_way_its_carrier_does(bound):
    b = _by_node(find_visual_bindings(bound))
    assert (b["SVJ::body::chassis"]["kind"], b["SVJ::body::chassis"]["part"]) == ("body", "chassis")
    assert b["SVJ::suspension::upright_fl"]["kind"] == "station"
    assert b["SVJ::suspension::upright_fl"]["station"] == "FL"
    assert b["SVJ::wheel::wheel_fl"]["spin"] is True
    assert b["SVJ::brake::disc_fl"]["spin"] is True
    assert b["SVJ::brake::caliper_fl"]["spin"] is False
    # §22.6 gives springs, dampers and ARBs no articulated placement, and a
    # rigid link is authored in the vehicle frame and not moved.
    assert b["SVJ::suspension::spring_fl"]["kind"] == "body"
    assert b["SVJ::suspension::lower_wishbone_fl"]["kind"] == "body"
    assert b["SVJ::steering::rack"]["kind"] == "body"
    assert b["SVJ::powertrain::engine"]["kind"] == "body"
    assert not any("issue" in x for x in b.values())


def test_link_between_points_carries_both_ends(bound):
    link = _by_node(find_visual_bindings(bound))["SVJ::suspension::upper_wishbone_fl"]
    inboard = bound["suspension"]["FL"]["topology"]["links"][0]["inboard_points"]
    assert link["kind"] == "link"
    assert link["outboard"] == "upper_ball_joint"
    # No from_point: the centroid of the inboard points, the pivot axis centre.
    assert link["inboard"] == pytest.approx([sum(p[i] for p in inboard) / 2 for i in range(3)])
    assert link["mesh_axis"] == "+y" and link["scale_to_length"] is False


def test_from_point_picks_one_inboard_point_and_range_is_checked(bound):
    links = bound["suspension"]["FL"]["topology"]["links"]
    links[0]["visual"]["from_point"] = 1
    link = _by_node(find_visual_bindings(bound))["SVJ::suspension::upper_wishbone_fl"]
    assert link["inboard"] == links[0]["inboard_points"][1]

    links[0]["visual"]["from_point"] = 5
    link = _by_node(find_visual_bindings(bound))["SVJ::suspension::upper_wishbone_fl"]
    assert "out of range" in link["issue"] and link["kind"] == "link"


def test_mass_body_is_not_offset_by_its_position(bound):
    bound["chassis"]["mass_bodies"] = [{"id": "engine_block", "mass": 150.0,
                                        "position": [0.4, 0.0, -0.45]}]
    _bind(bound, "chassis", "mass_bodies", 0, node="SVJ::body::engine_block")
    entry = _by_node(find_visual_bindings(bound))["SVJ::body::engine_block"]
    # Authored in the vehicle frame (§22.6 "rigid"): nothing for the viewer
    # to add, so no position travels with it.
    assert entry["kind"] == "body" and "position" not in entry


def test_mesh_ref_may_be_omitted_with_a_single_mesh(bound):
    del bound["chassis"]["visual"]["mesh_ref"]
    assert _by_node(find_visual_bindings(bound))["SVJ::body::chassis"]["mesh_ref"] == "car"

    bound["assets"]["meshes"].append({"id": "wheels", "uri": "wheels.glb"})
    chassis = _by_node(find_visual_bindings(bound))["SVJ::body::chassis"]
    assert "no mesh_ref" in chassis["issue"]


def test_link_between_points_elsewhere_is_reported_and_shown_rigid(bound):
    bound["chassis"]["visual"]["placement"] = "link_between_points"
    chassis = _by_node(find_visual_bindings(bound))["SVJ::body::chassis"]
    assert chassis["kind"] == "body" and "only valid on a link" in chassis["issue"]


def test_a_node_bound_twice_is_drawn_once(bound):
    _bind(bound, "suspension", "FR", "wheel", node="SVJ::wheel::wheel_fl")
    entries = [b for b in find_visual_bindings(bound) if b["node"] == "SVJ::wheel::wheel_fl"]
    assert len(entries) == 2
    assert "duplicate_of" not in entries[0]
    assert entries[1]["duplicate_of"] == ".suspension.FL.wheel" and "issue" in entries[1]


def test_a_shared_axle_body_is_not_a_duplicate_error(bound):
    for station in ("RL", "RR"):
        bound["suspension"][station]["topology"]["axle_body"] = {"type": "solid_axle"}
        _bind(bound, "suspension", station, "topology", "axle_body", node="SVJ::suspension::rear_axle")
    entries = [b for b in find_visual_bindings(bound) if b["node"] == "SVJ::suspension::rear_axle"]
    assert [e["part"] for e in entries] == ["axle_body", "axle_body"]
    assert entries[1]["duplicate_of"] and "issue" not in entries[1]


def test_v097_corner_level_binding_still_follows_the_upright(bound):
    bound["suspension"]["RR"]["visual"] = {"mesh_ref": "car", "node": "SVJ::body::upright_rr"}
    entry = _by_node(find_visual_bindings(bound))["SVJ::body::upright_rr"]
    assert (entry["kind"], entry["part"], entry["legacy_corner"]) == ("station", "upright", True)


def test_driveshaft_between_its_joints(bound):
    bound["powertrain"]["driveshafts"] = [{
        "id": "propshaft",
        "joint_front": {"type": "universal", "position": [-0.5, 0.0, -0.30]},
        "joint_rear": {"type": "universal", "position": [-2.3, 0.0, -0.28]},
        "visual": {"mesh_ref": "car", "node": "SVJ::powertrain::propshaft",
                   "placement": "link_between_points", "scale_to_length": True}}]
    shaft = _by_node(find_visual_bindings(bound))["SVJ::powertrain::propshaft"]
    assert shaft["kind"] == "segment"
    assert shaft["from"] == [-0.5, 0.0, -0.30] and shaft["to"] == [-2.3, 0.0, -0.28]
    assert shaft["mesh_axis"] == "+x" and shaft["scale_to_length"] is True
