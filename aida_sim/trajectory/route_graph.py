#!/usr/bin/env python3
"""
Graph-Based Route Planner for AIDA

Implements A* pathfinding on a navigation graph for flight route planning.

The graph consists of:
- Nodes: Airports, VORs, waypoints, intersections, user-defined fixes
- Edges: Airways, direct segments (weighted by cost)

Cost function considers:
- Distance
- Wind (headwind penalty, tailwind bonus)
- Airspace restrictions (penalties for certain areas)
- Terrain clearance requirements

Author: Kushal Koirala (with Claude Code)
Date: January 2026
"""

import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum


# Constants
NM_TO_FT = 6076.12
FT_TO_M = 0.3048
M_TO_FT = 3.28084
EARTH_RADIUS_NM = 3440.065


class NavaidType(Enum):
    """Types of navigation aids and waypoints."""
    AIRPORT = "airport"
    VOR = "vor"
    VORTAC = "vortac"
    NDB = "ndb"
    FIX = "fix"           # Named intersection
    WAYPOINT = "waypoint"  # GPS waypoint
    USER = "user"          # User-defined point


class AirwayType(Enum):
    """Types of airways."""
    VICTOR = "victor"      # Low altitude (below 18000 ft)
    JET = "jet"            # High altitude (above 18000 ft)
    DIRECT = "direct"      # Direct routing (no airway)
    SID = "sid"            # Standard Instrument Departure
    STAR = "star"          # Standard Terminal Arrival Route
    APPROACH = "approach"  # Instrument approach segment


@dataclass
class NavNode:
    """
    A node in the navigation graph.

    Represents any point that can be part of a route:
    airports, VORs, waypoints, fixes, etc.
    """
    id: str                      # Unique identifier (e.g., "KHUT", "ICT", "WEAVR")
    name: str                    # Full name
    lat: float                   # Latitude (degrees)
    lon: float                   # Longitude (degrees)
    elevation_ft: float          # Elevation MSL (feet)
    nav_type: NavaidType

    # Optional properties
    frequency: Optional[float] = None  # VOR/NDB frequency
    runway_heading: Optional[float] = None  # For airports, primary runway heading
    magnetic_variation: float = 5.0  # Local magnetic variation (degrees east)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, NavNode):
            return self.id == other.id
        return False


@dataclass
class NavEdge:
    """
    An edge in the navigation graph.

    Represents a connection between two nodes (airway segment or direct route).
    """
    from_node: str               # Source node ID
    to_node: str                 # Destination node ID
    airway_type: AirwayType
    airway_id: Optional[str] = None  # Airway identifier (e.g., "V4", "J80")

    # Altitude constraints
    min_altitude_ft: float = 0.0
    max_altitude_ft: float = 60000.0

    # Cost modifiers
    distance_nm: float = 0.0     # Will be computed from node positions
    base_cost: float = 0.0       # Base cost (usually distance)
    airspace_penalty: float = 0.0  # Additional cost for restricted airspace
    terrain_penalty: float = 0.0   # Additional cost for terrain considerations

    # Bidirectional flag
    bidirectional: bool = True

    def __hash__(self):
        return hash((self.from_node, self.to_node, self.airway_id))


@dataclass
class RouteSegment:
    """A segment of a computed route."""
    from_node: NavNode
    to_node: NavNode
    edge: NavEdge
    distance_nm: float
    heading_deg: float           # True heading
    altitude_ft: float           # Cruise altitude for this segment
    airspeed_kts: float          # Target airspeed


@dataclass
class Route:
    """A complete route from origin to destination."""
    origin: NavNode
    destination: NavNode
    segments: List[RouteSegment]
    total_distance_nm: float
    total_cost: float
    waypoints: List[NavNode]     # Ordered list of all waypoints

    def __str__(self) -> str:
        wp_str = " -> ".join([wp.id for wp in self.waypoints])
        return f"Route: {wp_str} ({self.total_distance_nm:.1f} NM)"


class NavigationGraph:
    """
    Navigation graph for route planning.

    Supports:
    - Adding nodes (airports, VORs, waypoints)
    - Adding edges (airways, direct routes)
    - A* pathfinding with customizable cost function
    - Automatic direct edge generation within range
    """

    def __init__(self):
        self.nodes: Dict[str, NavNode] = {}
        self.edges: Dict[str, List[NavEdge]] = {}  # Adjacency list
        self.reverse_edges: Dict[str, List[NavEdge]] = {}  # For bidirectional lookup

        # Default parameters
        self.direct_range_nm = 100.0  # Max range for auto-generated direct routes
        self.default_cruise_altitude = 5500.0
        self.default_airspeed_kts = 110.0

    def add_node(self, node: NavNode) -> None:
        """Add a navigation node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self.edges:
            self.edges[node.id] = []
        if node.id not in self.reverse_edges:
            self.reverse_edges[node.id] = []

    def add_edge(self, edge: NavEdge) -> None:
        """Add an edge to the graph."""
        # Compute distance if not provided
        if edge.distance_nm == 0.0:
            from_node = self.nodes.get(edge.from_node)
            to_node = self.nodes.get(edge.to_node)
            if from_node and to_node:
                edge.distance_nm = self._haversine_distance(
                    from_node.lat, from_node.lon,
                    to_node.lat, to_node.lon
                )

        # Set base cost if not provided
        if edge.base_cost == 0.0:
            edge.base_cost = edge.distance_nm

        # Add forward edge
        if edge.from_node in self.edges:
            self.edges[edge.from_node].append(edge)

        # Add reverse edge for bidirectional routes
        if edge.bidirectional:
            reverse_edge = NavEdge(
                from_node=edge.to_node,
                to_node=edge.from_node,
                airway_type=edge.airway_type,
                airway_id=edge.airway_id,
                min_altitude_ft=edge.min_altitude_ft,
                max_altitude_ft=edge.max_altitude_ft,
                distance_nm=edge.distance_nm,
                base_cost=edge.base_cost,
                airspace_penalty=edge.airspace_penalty,
                terrain_penalty=edge.terrain_penalty,
                bidirectional=False  # Avoid infinite recursion
            )
            if edge.to_node in self.edges:
                self.edges[edge.to_node].append(reverse_edge)

    def generate_direct_edges(self, max_range_nm: Optional[float] = None) -> int:
        """
        Generate direct edges between all nodes within range.

        Returns the number of edges created.
        """
        if max_range_nm is None:
            max_range_nm = self.direct_range_nm

        count = 0
        node_ids = list(self.nodes.keys())

        for i, id1 in enumerate(node_ids):
            for id2 in node_ids[i+1:]:
                node1 = self.nodes[id1]
                node2 = self.nodes[id2]

                dist = self._haversine_distance(
                    node1.lat, node1.lon,
                    node2.lat, node2.lon
                )

                if dist <= max_range_nm:
                    # Check if edge already exists
                    existing = any(
                        e.to_node == id2 for e in self.edges.get(id1, [])
                    )
                    if not existing:
                        edge = NavEdge(
                            from_node=id1,
                            to_node=id2,
                            airway_type=AirwayType.DIRECT,
                            distance_nm=dist,
                            base_cost=dist,
                            bidirectional=True
                        )
                        self.add_edge(edge)
                        count += 1

        return count

    def find_route(self,
                   origin_id: str,
                   destination_id: str,
                   cruise_altitude_ft: Optional[float] = None,
                   wind_direction_deg: float = 0.0,
                   wind_speed_kts: float = 0.0) -> Optional[Route]:
        """
        Find optimal route using A* algorithm.

        Args:
            origin_id: Starting node ID
            destination_id: Ending node ID
            cruise_altitude_ft: Desired cruise altitude
            wind_direction_deg: Wind direction (from) in degrees true
            wind_speed_kts: Wind speed in knots

        Returns:
            Route object or None if no path found
        """
        if origin_id not in self.nodes or destination_id not in self.nodes:
            return None

        origin = self.nodes[origin_id]
        destination = self.nodes[destination_id]

        if cruise_altitude_ft is None:
            cruise_altitude_ft = self.default_cruise_altitude

        # A* algorithm
        # Priority queue: (f_score, counter, node_id, path)
        counter = 0
        open_set = [(0, counter, origin_id, [origin_id])]
        heapq.heapify(open_set)

        # g_score: cost from start to node
        g_score: Dict[str, float] = {origin_id: 0}

        # Closed set
        closed_set: Set[str] = set()

        while open_set:
            _, _, current_id, path = heapq.heappop(open_set)

            if current_id in closed_set:
                continue

            if current_id == destination_id:
                # Found path - construct route
                return self._construct_route(
                    path, cruise_altitude_ft,
                    wind_direction_deg, wind_speed_kts
                )

            closed_set.add(current_id)
            current_node = self.nodes[current_id]

            # Explore neighbors
            for edge in self.edges.get(current_id, []):
                neighbor_id = edge.to_node

                if neighbor_id in closed_set:
                    continue

                # Check altitude constraints
                if cruise_altitude_ft < edge.min_altitude_ft or \
                   cruise_altitude_ft > edge.max_altitude_ft:
                    continue

                # Calculate edge cost with wind
                edge_cost = self._calculate_edge_cost(
                    edge, current_node, self.nodes[neighbor_id],
                    wind_direction_deg, wind_speed_kts
                )

                tentative_g = g_score[current_id] + edge_cost

                if neighbor_id not in g_score or tentative_g < g_score[neighbor_id]:
                    g_score[neighbor_id] = tentative_g

                    # Heuristic: straight-line distance to destination
                    h = self._haversine_distance(
                        self.nodes[neighbor_id].lat,
                        self.nodes[neighbor_id].lon,
                        destination.lat,
                        destination.lon
                    )

                    f_score = tentative_g + h
                    counter += 1
                    new_path = path + [neighbor_id]
                    heapq.heappush(open_set, (f_score, counter, neighbor_id, new_path))

        return None  # No path found

    def _calculate_edge_cost(self,
                             edge: NavEdge,
                             from_node: NavNode,
                             to_node: NavNode,
                             wind_dir_deg: float,
                             wind_speed_kts: float) -> float:
        """
        Calculate the cost of traversing an edge.

        Includes:
        - Base distance cost
        - Wind component (headwind penalty, tailwind bonus)
        - Airspace and terrain penalties
        """
        # Base cost is distance
        cost = edge.base_cost

        # Calculate track heading
        track_heading = self._bearing(
            from_node.lat, from_node.lon,
            to_node.lat, to_node.lon
        )

        # Wind component
        # Headwind is positive (adds to cost), tailwind is negative (reduces cost)
        if wind_speed_kts > 0:
            # Wind effect angle (angle between wind direction and track)
            # Wind direction is "from", so we compare with track heading
            wind_effect_angle = np.radians(wind_dir_deg - track_heading)
            headwind_component = wind_speed_kts * np.cos(wind_effect_angle)

            # Convert to time penalty/bonus
            # Assuming 110 kts groundspeed baseline
            groundspeed = self.default_airspeed_kts - headwind_component
            if groundspeed > 20:  # Sanity check
                time_factor = self.default_airspeed_kts / groundspeed
                cost *= time_factor

        # Add penalties
        cost += edge.airspace_penalty
        cost += edge.terrain_penalty

        return cost

    def _construct_route(self,
                         path: List[str],
                         cruise_altitude_ft: float,
                         wind_dir_deg: float,
                         wind_speed_kts: float) -> Route:
        """Construct a Route object from a path of node IDs."""
        waypoints = [self.nodes[node_id] for node_id in path]
        segments = []
        total_distance = 0.0
        total_cost = 0.0

        for i in range(len(path) - 1):
            from_id = path[i]
            to_id = path[i + 1]
            from_node = self.nodes[from_id]
            to_node = self.nodes[to_id]

            # Find the edge
            edge = None
            for e in self.edges.get(from_id, []):
                if e.to_node == to_id:
                    edge = e
                    break

            if edge is None:
                # Create a direct edge if none exists
                dist = self._haversine_distance(
                    from_node.lat, from_node.lon,
                    to_node.lat, to_node.lon
                )
                edge = NavEdge(
                    from_node=from_id,
                    to_node=to_id,
                    airway_type=AirwayType.DIRECT,
                    distance_nm=dist,
                    base_cost=dist
                )

            heading = self._bearing(
                from_node.lat, from_node.lon,
                to_node.lat, to_node.lon
            )

            segment = RouteSegment(
                from_node=from_node,
                to_node=to_node,
                edge=edge,
                distance_nm=edge.distance_nm,
                heading_deg=heading,
                altitude_ft=cruise_altitude_ft,
                airspeed_kts=self.default_airspeed_kts
            )
            segments.append(segment)
            total_distance += edge.distance_nm
            total_cost += self._calculate_edge_cost(
                edge, from_node, to_node, wind_dir_deg, wind_speed_kts
            )

        return Route(
            origin=waypoints[0],
            destination=waypoints[-1],
            segments=segments,
            total_distance_nm=total_distance,
            total_cost=total_cost,
            waypoints=waypoints
        )

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float,
                            lat2: float, lon2: float) -> float:
        """Calculate great-circle distance in nautical miles."""
        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2, lon2 = np.radians(lat2), np.radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        return EARTH_RADIUS_NM * c

    @staticmethod
    def _bearing(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
        """Calculate initial bearing in degrees true."""
        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2, lon2 = np.radians(lat2), np.radians(lon2)

        dlon = lon2 - lon1

        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

        bearing = np.degrees(np.arctan2(x, y))
        return (bearing + 360) % 360


def create_kansas_navgraph() -> NavigationGraph:
    """
    Create a navigation graph for the Kansas/Wichita region.

    Includes:
    - Airports: SN65, KHUT, KICT, KAAO
    - VORs: ICT, HUT
    - Fixes/Waypoints: Common GPS waypoints
    """
    graph = NavigationGraph()

    # === AIRPORTS ===

    # SN65 - Lake Waltanna (origin)
    graph.add_node(NavNode(
        id="SN65",
        name="Lake Waltanna Airport",
        lat=37.594167,
        lon=-97.615833,
        elevation_ft=1448.0,
        nav_type=NavaidType.AIRPORT,
        runway_heading=4.0  # Runway 35 heading
    ))

    # KHUT - Hutchinson Regional (destination)
    graph.add_node(NavNode(
        id="KHUT",
        name="Hutchinson Regional Airport",
        lat=38.066167,
        lon=-97.860500,
        elevation_ft=1542.0,
        nav_type=NavaidType.AIRPORT,
        runway_heading=314.0  # Runway 31 heading
    ))

    # KICT - Wichita Eisenhower (major reference)
    graph.add_node(NavNode(
        id="KICT",
        name="Wichita Eisenhower National",
        lat=37.649944,
        lon=-97.433056,
        elevation_ft=1333.0,
        nav_type=NavaidType.AIRPORT,
        runway_heading=14.0  # Runway 01L/R heading
    ))

    # KAAO - Colonel James Jabara
    graph.add_node(NavNode(
        id="KAAO",
        name="Colonel James Jabara Airport",
        lat=37.747500,
        lon=-97.221389,
        elevation_ft=1421.0,
        nav_type=NavaidType.AIRPORT,
        runway_heading=180.0
    ))

    # === VORs ===

    # ICT VOR/DME
    graph.add_node(NavNode(
        id="ICT",
        name="Wichita VOR/DME",
        lat=37.648333,
        lon=-97.418333,
        elevation_ft=1340.0,
        nav_type=NavaidType.VOR,
        frequency=113.8
    ))

    # HUT VOR (near Hutchinson)
    graph.add_node(NavNode(
        id="HUT",
        name="Hutchinson VOR",
        lat=38.069444,
        lon=-97.860556,
        elevation_ft=1550.0,
        nav_type=NavaidType.VOR,
        frequency=116.8
    ))

    # === FIXES/WAYPOINTS ===

    # WEAVR - common GPS fix between SN65 and KHUT
    graph.add_node(NavNode(
        id="WEAVR",
        name="WEAVR Intersection",
        lat=37.85,
        lon=-97.75,
        elevation_ft=1500.0,
        nav_type=NavaidType.FIX
    ))

    # GNDDY - south of Hutchinson
    graph.add_node(NavNode(
        id="GNDDY",
        name="GNDDY Waypoint",
        lat=37.95,
        lon=-97.80,
        elevation_ft=1520.0,
        nav_type=NavaidType.WAYPOINT
    ))

    # BRWNZ - west of ICT
    graph.add_node(NavNode(
        id="BRWNZ",
        name="BRWNZ Waypoint",
        lat=37.70,
        lon=-97.60,
        elevation_ft=1450.0,
        nav_type=NavaidType.WAYPOINT
    ))

    # === APPROACH FIXES FOR KHUT ===

    # KHUT IAF (Initial Approach Fix) - 10nm from runway on extended centerline
    iaf_lat = 38.066167 + (10 * np.cos(np.radians(314 + 180))) / 60  # Approx
    iaf_lon = -97.860500 + (10 * np.sin(np.radians(314 + 180))) / (60 * np.cos(np.radians(38.066167)))
    graph.add_node(NavNode(
        id="HUTIA",
        name="KHUT IAF",
        lat=37.95,  # ~10nm SE of KHUT on runway 31 extended
        lon=-97.72,
        elevation_ft=1600.0,
        nav_type=NavaidType.FIX
    ))

    # KHUT FAF (Final Approach Fix) - 5nm from runway
    graph.add_node(NavNode(
        id="HUTFA",
        name="KHUT FAF",
        lat=38.01,  # ~5nm SE of KHUT
        lon=-97.79,
        elevation_ft=1550.0,
        nav_type=NavaidType.FIX
    ))

    # === EDGES (Airways and direct routes) ===

    # Generate direct edges between all nodes within range
    # This creates a connected graph
    graph.generate_direct_edges(max_range_nm=50.0)

    # Add specific airway segments if needed
    # Victor airways would go here

    # Add approach segments with altitude constraints
    graph.add_edge(NavEdge(
        from_node="HUTIA",
        to_node="HUTFA",
        airway_type=AirwayType.APPROACH,
        airway_id="ILS31",
        min_altitude_ft=2500.0,
        max_altitude_ft=6000.0
    ))

    graph.add_edge(NavEdge(
        from_node="HUTFA",
        to_node="KHUT",
        airway_type=AirwayType.APPROACH,
        airway_id="ILS31",
        min_altitude_ft=1542.0,  # Field elevation
        max_altitude_ft=3000.0
    ))

    return graph


def route_to_waypoints(route: Route) -> List[dict]:
    """
    Convert a Route to a list of waypoint dictionaries.

    Returns waypoints in format suitable for trajectory planner.
    """
    waypoints = []

    for i, node in enumerate(route.waypoints):
        wp = {
            'id': node.id,
            'name': node.name,
            'lat': node.lat,
            'lon': node.lon,
            'elevation_ft': node.elevation_ft,
            'nav_type': node.nav_type.value,
        }

        # Add segment info if not the last waypoint
        if i < len(route.segments):
            seg = route.segments[i]
            wp['heading_deg'] = seg.heading_deg
            wp['altitude_ft'] = seg.altitude_ft
            wp['airspeed_kts'] = seg.airspeed_kts
            wp['distance_nm'] = seg.distance_nm

        waypoints.append(wp)

    return waypoints


class FlightPlanGenerator:
    """
    Generates complete flight plans from A* routes.

    Takes a route and generates a detailed flight plan with:
    - Departure procedure (takeoff, initial climb)
    - Enroute phase (cruise waypoints)
    - Arrival procedure (descent, approach, landing)
    """

    def __init__(self, graph: NavigationGraph):
        self.graph = graph

        # Performance parameters
        self.climb_rate_fpm = 500.0
        self.descent_rate_fpm = 500.0
        self.cruise_speed_kts = 110.0
        self.approach_speed_kts = 85.0
        self.climb_speed_kts = 74.0
        self.glideslope_deg = 3.0

    def generate_flight_plan(self,
                             origin_id: str,
                             destination_id: str,
                             cruise_altitude_ft: float = 5500.0,
                             departure_runway: Optional[str] = None,
                             arrival_runway: Optional[str] = None) -> Optional[List[dict]]:
        """
        Generate a complete flight plan.

        Returns list of waypoint dictionaries with:
        - position (lat, lon)
        - altitude
        - speed
        - segment type (departure, enroute, arrival)
        """
        # Get the base route
        route = self.graph.find_route(origin_id, destination_id, cruise_altitude_ft)
        if not route:
            return None

        origin = self.graph.nodes[origin_id]
        destination = self.graph.nodes[destination_id]

        flight_plan = []

        # === DEPARTURE PHASE ===
        departure_hdg = origin.runway_heading or 0.0

        # 1. Takeoff position
        flight_plan.append({
            'id': f'{origin_id}_RWY',
            'name': f'{origin.name} Runway',
            'lat': origin.lat,
            'lon': origin.lon,
            'altitude_ft': origin.elevation_ft,
            'speed_kts': 0.0,
            'segment': 'TAKEOFF',
            'heading_deg': departure_hdg,
        })

        # 2. Liftoff point (1000ft down runway)
        liftoff_lat, liftoff_lon = self._project_point(
            origin.lat, origin.lon, departure_hdg, 1000 / NM_TO_FT
        )
        flight_plan.append({
            'id': f'{origin_id}_LIFT',
            'name': 'Liftoff',
            'lat': liftoff_lat,
            'lon': liftoff_lon,
            'altitude_ft': origin.elevation_ft + 50,
            'speed_kts': 60.0,
            'segment': 'TAKEOFF',
            'heading_deg': departure_hdg,
        })

        # 3. Initial climb (1nm from airport)
        climb1_lat, climb1_lon = self._project_point(
            origin.lat, origin.lon, departure_hdg, 1.0
        )
        flight_plan.append({
            'id': f'{origin_id}_CLB1',
            'name': 'Initial Climb',
            'lat': climb1_lat,
            'lon': climb1_lon,
            'altitude_ft': origin.elevation_ft + 500,
            'speed_kts': self.climb_speed_kts,
            'segment': 'CLIMB',
            'heading_deg': departure_hdg,
        })

        # 4. Turn to course and continue climb (2nm from airport)
        # Calculate heading to first enroute waypoint
        if len(route.waypoints) > 1:
            next_wp = route.waypoints[1]
            course_hdg = self._bearing(origin.lat, origin.lon, next_wp.lat, next_wp.lon)
        else:
            course_hdg = departure_hdg

        climb2_lat, climb2_lon = self._project_point(
            origin.lat, origin.lon, course_hdg, 2.0
        )
        flight_plan.append({
            'id': f'{origin_id}_CLB2',
            'name': 'Climbing Turn',
            'lat': climb2_lat,
            'lon': climb2_lon,
            'altitude_ft': origin.elevation_ft + 1500,
            'speed_kts': self.climb_speed_kts,
            'segment': 'CLIMB',
            'heading_deg': course_hdg,
        })

        # 5. Top of Climb (5nm from airport or when reaching cruise altitude)
        toc_lat, toc_lon = self._project_point(
            origin.lat, origin.lon, course_hdg, 5.0
        )
        flight_plan.append({
            'id': 'TOC',
            'name': 'Top of Climb',
            'lat': toc_lat,
            'lon': toc_lon,
            'altitude_ft': cruise_altitude_ft,
            'speed_kts': self.cruise_speed_kts,
            'segment': 'CRUISE',
            'heading_deg': course_hdg,
        })

        # === ENROUTE PHASE ===
        # Add intermediate waypoints from the route (skip origin and destination)
        for wp in route.waypoints[1:-1]:
            hdg_to_next = self._bearing(
                wp.lat, wp.lon,
                destination.lat, destination.lon
            )
            flight_plan.append({
                'id': wp.id,
                'name': wp.name,
                'lat': wp.lat,
                'lon': wp.lon,
                'altitude_ft': cruise_altitude_ft,
                'speed_kts': self.cruise_speed_kts,
                'segment': 'CRUISE',
                'heading_deg': hdg_to_next,
            })

        # === ARRIVAL PHASE ===
        arrival_hdg = destination.runway_heading or 0.0
        final_hdg = arrival_hdg  # The runway heading we're landing on

        # Calculate approach course from runway heading
        # Runway 31 means landing heading 314, approach from 134
        approach_inbound = final_hdg

        # Get the last cruise waypoint position
        last_cruise = flight_plan[-1]
        last_cruise_lat = last_cruise['lat']
        last_cruise_lon = last_cruise['lon']

        # Calculate direct heading from last cruise point to destination
        direct_to_dest = self._bearing(last_cruise_lat, last_cruise_lon,
                                       destination.lat, destination.lon)

        # Distance from last cruise point to destination
        dist_to_dest = self._haversine(last_cruise_lat, last_cruise_lon,
                                       destination.lat, destination.lon)

        # Position TOD at ~10nm from destination, but along the CURRENT track
        # This keeps the aircraft on a continuous path until it needs to turn final
        tod_dist_from_dest = 10.0
        tod_dist_from_cruise = max(0.0, dist_to_dest - tod_dist_from_dest)

        if tod_dist_from_cruise > 1.0:
            # TOD is positioned along the cruise track, not on the approach course
            tod_lat, tod_lon = self._project_point(
                last_cruise_lat, last_cruise_lon, direct_to_dest, tod_dist_from_cruise
            )
        else:
            # Very close to destination, put TOD 10nm back on approach course
            tod_lat, tod_lon = self._project_point(
                destination.lat, destination.lon,
                (approach_inbound + 180) % 360,
                10.0
            )

        # The approach needs to smoothly transition from cruise heading to final approach course
        # We'll fly past the destination, then turn back for final approach (teardrop pattern)

        # 1. Position TOD where we start thinking about approach
        # Keep flying on cruise heading until close to destination
        flight_plan.append({
            'id': 'TOD',
            'name': 'Top of Descent',
            'lat': tod_lat,
            'lon': tod_lon,
            'altitude_ft': cruise_altitude_ft,
            'speed_kts': self.cruise_speed_kts,
            'segment': 'DESCENT',
            'heading_deg': direct_to_dest,
        })

        # Calculate the heading difference between cruise track and final approach
        heading_diff = abs(direct_to_dest - approach_inbound)
        if heading_diff > 180:
            heading_diff = 360 - heading_diff

        # 2. For a straight-in approach (within 30° of final), just descend directly
        #    Otherwise, we need an approach pattern
        if heading_diff < 30:
            # Straight-in approach - fly direct to extended final
            faf_dist_nm = 5.0
            faf_alt = destination.elevation_ft + (faf_dist_nm * NM_TO_FT * np.tan(np.radians(self.glideslope_deg)))
            faf_lat, faf_lon = self._project_point(
                destination.lat, destination.lon,
                (approach_inbound + 180) % 360,
                faf_dist_nm
            )
            flight_plan.append({
                'id': 'FAF',
                'name': 'Final Approach Fix',
                'lat': faf_lat,
                'lon': faf_lon,
                'altitude_ft': faf_alt,
                'speed_kts': self.approach_speed_kts,
                'segment': 'APPROACH',
                'heading_deg': approach_inbound,
            })
        else:
            # Need a procedure turn or base turn
            # Use a simple base leg pattern: continue past, then turn to base, then turn final

            # Continue toward destination on cruise heading until we're abeam the runway
            # Then turn to a downwind/base position

            # Position base turn point roughly 3nm to the side of the runway
            # Determine which side to turn (left or right traffic)
            cross_track = (direct_to_dest - approach_inbound + 360) % 360
            if cross_track > 180:
                # Turn right to final (left traffic pattern)
                base_offset_heading = (approach_inbound + 90) % 360
            else:
                # Turn left to final (right traffic pattern)
                base_offset_heading = (approach_inbound - 90 + 360) % 360

            # Base point: 3nm from runway threshold, perpendicular to approach course
            base_dist_nm = 3.0
            base_lat, base_lon = self._project_point(
                destination.lat, destination.lon,
                base_offset_heading,
                base_dist_nm
            )
            # Also extend back along the approach course from there
            base_lat, base_lon = self._project_point(
                base_lat, base_lon,
                (approach_inbound + 180) % 360,  # Back from runway
                3.0  # 3nm back
            )

            base_alt = destination.elevation_ft + 1500  # Pattern altitude

            # Heading at base: pointing toward final approach course
            hdg_at_base = self._bearing(base_lat, base_lon, destination.lat, destination.lon)

            flight_plan.append({
                'id': 'BASE',
                'name': 'Base Turn',
                'lat': base_lat,
                'lon': base_lon,
                'altitude_ft': base_alt,
                'speed_kts': self.approach_speed_kts,
                'segment': 'APPROACH',
                'heading_deg': hdg_at_base,
            })

            # FAF at 5nm
            faf_dist_nm = 5.0
            faf_alt = destination.elevation_ft + (faf_dist_nm * NM_TO_FT * np.tan(np.radians(self.glideslope_deg)))
            faf_lat, faf_lon = self._project_point(
                destination.lat, destination.lon,
                (approach_inbound + 180) % 360,
                faf_dist_nm
            )
            flight_plan.append({
                'id': 'FAF',
                'name': 'Final Approach Fix',
                'lat': faf_lat,
                'lon': faf_lon,
                'altitude_ft': faf_alt,
                'speed_kts': self.approach_speed_kts,
                'segment': 'APPROACH',
                'heading_deg': approach_inbound,
            })

        # 3. Additional glideslope points (3nm, 2nm, 1nm)
        for dist_nm in [3.0, 2.0, 1.0]:
            alt = destination.elevation_ft + (dist_nm * NM_TO_FT * np.tan(np.radians(self.glideslope_deg)))
            lat, lon = self._project_point(
                destination.lat, destination.lon,
                (approach_inbound + 180) % 360,
                dist_nm
            )
            flight_plan.append({
                'id': f'FIN{int(dist_nm)}',
                'name': f'{dist_nm:.0f}nm Final',
                'lat': lat,
                'lon': lon,
                'altitude_ft': alt,
                'speed_kts': self.approach_speed_kts,
                'segment': 'LANDING',
                'heading_deg': approach_inbound,
            })

        # 5. Threshold (50ft AGL)
        flight_plan.append({
            'id': 'THR',
            'name': 'Threshold',
            'lat': destination.lat,
            'lon': destination.lon,
            'altitude_ft': destination.elevation_ft + 50,
            'speed_kts': 65.0,
            'segment': 'LANDING',
            'heading_deg': approach_inbound,
        })

        # 6. Touchdown (300ft past threshold)
        td_lat, td_lon = self._project_point(
            destination.lat, destination.lon,
            approach_inbound,
            300 / NM_TO_FT
        )
        flight_plan.append({
            'id': 'TD',
            'name': 'Touchdown',
            'lat': td_lat,
            'lon': td_lon,
            'altitude_ft': destination.elevation_ft,
            'speed_kts': 50.0,
            'segment': 'LANDING',
            'heading_deg': approach_inbound,
        })

        return flight_plan

    def _project_point(self, lat: float, lon: float,
                       heading_deg: float, distance_nm: float) -> Tuple[float, float]:
        """Project a point from lat/lon along a heading for a distance."""
        # Simple approximation for short distances
        lat_rad = np.radians(lat)
        heading_rad = np.radians(heading_deg)

        # Convert distance to degrees
        # 1 degree latitude ≈ 60 nm
        # 1 degree longitude ≈ 60 nm * cos(lat)
        dlat = distance_nm / 60.0 * np.cos(heading_rad)
        dlon = distance_nm / 60.0 * np.sin(heading_rad) / np.cos(lat_rad)

        return lat + dlat, lon + dlon

    def _bearing(self, lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
        """Calculate initial bearing in degrees true."""
        return NavigationGraph._bearing(lat1, lon1, lat2, lon2)

    def _haversine(self, lat1: float, lon1: float,
                   lat2: float, lon2: float) -> float:
        """Calculate great-circle distance in nautical miles."""
        return NavigationGraph._haversine_distance(lat1, lon1, lat2, lon2)


if __name__ == "__main__":
    print("Navigation Graph Route Planner")
    print("=" * 60)

    # Create the Kansas navigation graph
    graph = create_kansas_navgraph()

    print(f"Graph has {len(graph.nodes)} nodes:")
    for node_id, node in graph.nodes.items():
        print(f"  {node_id}: {node.name} ({node.nav_type.value})")

    print(f"\nTotal edges: {sum(len(edges) for edges in graph.edges.values())}")

    # Find route from SN65 to KHUT
    print("\n" + "=" * 60)
    print("Finding route: SN65 -> KHUT")

    route = graph.find_route("SN65", "KHUT", cruise_altitude_ft=5500.0)

    if route:
        print(f"\n{route}")
        print(f"Total cost: {route.total_cost:.1f}")

    # Generate complete flight plan
    print("\n" + "=" * 60)
    print("Generating complete flight plan...")

    planner = FlightPlanGenerator(graph)
    flight_plan = planner.generate_flight_plan("SN65", "KHUT", cruise_altitude_ft=5500.0)

    if flight_plan:
        print(f"\nFlight Plan: {len(flight_plan)} waypoints")
        print("-" * 70)
        print(f"{'#':<3} {'ID':<12} {'Segment':<10} {'Alt(ft)':<8} {'Spd(kts)':<8} {'Hdg':<6} {'Lat':<10} {'Lon':<10}")
        print("-" * 70)
        for i, wp in enumerate(flight_plan):
            print(f"{i+1:<3} {wp['id']:<12} {wp['segment']:<10} {wp['altitude_ft']:<8.0f} {wp['speed_kts']:<8.0f} {wp['heading_deg']:<6.0f} {wp['lat']:<10.4f} {wp['lon']:<10.4f}")
    else:
        print("Failed to generate flight plan!")
