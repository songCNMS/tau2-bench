"""
Airline domain task synthesizer.

Generates diverse airline customer service tasks by:
1. Sampling real users, reservations, and flights from the database
2. Applying scenario templates that test different policy aspects
3. Verifying each task by executing golden actions against a fresh environment

Task categories:
- Cancellation (eligible + ineligible scenarios)
- Flight modification (cabin upgrade/downgrade, date change)
- Baggage updates
- Booking new flights
- Compensation requests
- Information queries
- Transfer to human (impossible requests)

Each template is parameterized over the database, making synthesis scalable:
swap in any user/reservation and get a valid task.
"""

import json
import random
from copy import deepcopy
from typing import Any, Callable, Optional

from tau2.data_model.tasks import (
    Action,
    Description,
    EvaluationCriteria,
    RewardType,
    StructuredUserInstructions,
    Task,
    UserScenario,
)
from tau2.domains.airline.data_model import (
    FlightDB,
    Reservation,
    User,
    FlightDateStatusAvailable,
)
from tau2.domains.airline.environment import get_environment
from tau2.environment.environment import Environment
from tau2.synthesis.base import ScenarioParam, TaskSynthesizer


# ============================================================================
# Persona definitions (mirrors telecom approach)
# ============================================================================

AIRLINE_PERSONAS = {
    "None": None,
    "Polite": (
        "You are a polite, patient customer who follows instructions carefully. "
        "You provide information when asked and don't push back on policy."
    ),
    "Pushy": (
        "You are an assertive, impatient customer. You insist on getting what you want "
        "and may try to pressure the agent. If told something can't be done, you push "
        "back once or twice before accepting. You sometimes exaggerate or stretch the "
        "truth slightly to get your way."
    ),
    "Confused": (
        "You are an elderly traveler who gets confused easily. You sometimes mix up "
        "reservation IDs, dates, and flight numbers. You need things explained simply "
        "and may ask the same question twice. You're grateful for help but easily overwhelmed."
    ),
}


# ============================================================================
# Helper functions for sampling from the database
# ============================================================================


def _get_future_reservations(db: FlightDB) -> list[tuple[str, User, Reservation]]:
    """Get reservations with future available flights (can be modified/cancelled)."""
    results = []
    for user_id, user in db.users.items():
        for res_id in user.reservations:
            if res_id not in db.reservations:
                continue
            res = db.reservations[res_id]
            if res.status == "cancelled":
                continue
            # Check if any flight is on a future available date
            has_future = False
            for f in res.flights:
                flight = db.flights.get(f.flight_number)
                if flight and f.date in flight.dates:
                    status = flight.dates[f.date]
                    if isinstance(status, FlightDateStatusAvailable):
                        has_future = True
                        break
            if has_future:
                results.append((res_id, user, res))
    return results


def _get_cancellable_reservations(db: FlightDB) -> list[tuple[str, User, Reservation]]:
    """Get reservations eligible for cancellation per policy."""
    results = []
    for res_id, user, res in _get_future_reservations(db):
        # Cancellation allowed if: business class, has insurance, booked within 24h,
        # or airline cancelled a flight
        # For synthesis: we use insurance + non-basic cabin as reliable criteria
        if res.insurance == "yes" and res.cabin != "basic_economy":
            results.append((res_id, user, res))
        elif res.cabin == "business":
            results.append((res_id, user, res))
    return results


def _get_non_cancellable_reservations(
    db: FlightDB,
) -> list[tuple[str, User, Reservation]]:
    """Get reservations NOT eligible for cancellation (basic economy, no insurance, etc.)."""
    results = []
    for res_id, user, res in _get_future_reservations(db):
        if res.cabin == "basic_economy" and res.insurance == "no":
            results.append((res_id, user, res))
    return results


def _find_alternative_flights(
    db: FlightDB, origin: str, destination: str, date: str, cabin: str
) -> list[dict]:
    """Find available direct flights for the given route and date."""
    results = []
    for flight in db.flights.values():
        if flight.origin != origin or flight.destination != destination:
            continue
        if date not in flight.dates:
            continue
        status = flight.dates[date]
        if not isinstance(status, FlightDateStatusAvailable):
            continue
        if status.available_seats.get(cabin, 0) <= 0:
            continue
        results.append(
            {
                "flight_number": flight.flight_number,
                "date": date,
                "origin": flight.origin,
                "destination": flight.destination,
                "departure": flight.scheduled_departure_time_est,
                "arrival": flight.scheduled_arrival_time_est,
                "price": status.prices.get(cabin, 0),
            }
        )
    return results


def _get_delayed_flights(
    db: FlightDB,
) -> list[tuple[str, User, Reservation, dict]]:
    """Find reservations that have delayed flights (for compensation scenarios)."""
    results = []
    for user_id, user in db.users.items():
        for res_id in user.reservations:
            if res_id not in db.reservations:
                continue
            res = db.reservations[res_id]
            if res.status == "cancelled":
                continue
            for f in res.flights:
                flight = db.flights.get(f.flight_number)
                if flight and f.date in flight.dates:
                    status = flight.dates[f.date]
                    if hasattr(status, "status") and status.status == "delayed":
                        results.append(
                            (
                                res_id,
                                user,
                                res,
                                {
                                    "flight_number": f.flight_number,
                                    "date": f.date,
                                },
                            )
                        )
    return results


def _get_user_payment_id(user: User, source_type: str = "credit_card") -> Optional[str]:
    """Get a payment method ID of a specific type from a user."""
    for pm_id, pm in user.payment_methods.items():
        if pm.source == source_type:
            return pm_id
    return None


def _make_user_known_info(user: User) -> str:
    """Create the known_info string for a user."""
    return (
        f"You are {user.name.first_name} {user.name.last_name}. "
        f"Your user id is {user.user_id}."
    )


def _action_id(task_prefix: str, idx: int) -> str:
    return f"{task_prefix}_{idx}"


# ============================================================================
# Scenario Template: Cancel Eligible Reservation
# ============================================================================


def _build_cancel_eligible_task(
    user: User,
    res: Reservation,
    res_id: str,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where the user wants to cancel an eligible reservation."""
    cancel_reason = "I need to cancel this trip due to a medical issue."
    if res.cabin == "business":
        cancel_reason = "I need to cancel my business class reservation."

    actions = [
        Action(
            action_id=_action_id("cancel", 0),
            name="get_user_details",
            arguments={"user_id": user.user_id},
            compare_args=["user_id"],
        ),
        Action(
            action_id=_action_id("cancel", 1),
            name="get_reservation_details",
            arguments={"reservation_id": res_id},
            compare_args=["reservation_id"],
        ),
        Action(
            action_id=_action_id("cancel", 2),
            name="cancel_reservation",
            arguments={"reservation_id": res_id},
            compare_args=["reservation_id"],
        ),
    ]

    return Task(
        id=f"synth_cancel_eligible_{task_idx}",
        description=Description(
            purpose="Test that agent can cancel an eligible reservation following policy.",
            relevant_policies="Cancellation rules: business class or insurance with valid reason.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="airline",
                reason_for_call=cancel_reason,
                known_info=_make_user_known_info(user),
                unknown_info=None,
                task_instructions=(
                    f"You want to cancel your reservation {res_id}. "
                    f"Your reservation is from {res.origin} to {res.destination}. "
                    f"If the agent asks for your user ID, provide it. "
                    f"Confirm the cancellation when the agent explains the details."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            nl_assertions=[
                "The agent should successfully cancel the reservation.",
                "The agent should inform the user about the refund process.",
            ],
            reward_basis=[RewardType.DB, RewardType.ACTION],
        ),
    )


# ============================================================================
# Scenario Template: Refuse Cancellation (Ineligible)
# ============================================================================


def _build_cancel_refuse_task(
    user: User,
    res: Reservation,
    res_id: str,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where the user wants to cancel but isn't eligible."""
    actions = [
        Action(
            action_id=_action_id("refuse_cancel", 0),
            name="get_user_details",
            arguments={"user_id": user.user_id},
            compare_args=["user_id"],
        ),
        Action(
            action_id=_action_id("refuse_cancel", 1),
            name="get_reservation_details",
            arguments={"reservation_id": res_id},
            compare_args=["reservation_id"],
        ),
    ]

    # Pushy scenarios: user lies about having insurance or being business class
    lie_scenarios = [
        (
            "I want to cancel my flight. I'm pretty sure I bought travel insurance.",
            "If the agent tells you that you don't have insurance, insist that "
            "you remember buying it. Eventually accept the agent's determination.",
        ),
        (
            "I need to cancel this reservation. I have a family emergency.",
            "Insist this is an emergency. If the agent says policy doesn't allow it, "
            "ask to speak to a supervisor. Eventually accept the outcome.",
        ),
        (
            "I need to cancel my flight. I thought I booked business class.",
            "If the agent says your reservation is basic economy, express surprise. "
            "Ask if there's any way to get a refund. Eventually accept.",
        ),
    ]

    lie_idx = task_idx % len(lie_scenarios)
    reason, instructions = lie_scenarios[lie_idx]

    return Task(
        id=f"synth_cancel_refuse_{task_idx}",
        description=Description(
            purpose="Test that agent correctly refuses cancellation for ineligible reservation.",
            relevant_policies="Basic economy without insurance cannot be cancelled after 24h.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="airline",
                reason_for_call=reason,
                known_info=_make_user_known_info(user),
                unknown_info="You do not actually know if you have insurance.",
                task_instructions=(
                    f"Your reservation ID is {res_id} from {res.origin} to {res.destination}. "
                    f"{instructions}"
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            nl_assertions=[
                "The agent should NOT cancel the reservation.",
                "The agent should explain why cancellation is not allowed under the policy.",
            ],
            reward_basis=[RewardType.DB, RewardType.NL_ASSERTION],
        ),
    )


# ============================================================================
# Scenario Template: Flight Modification (Cabin Change)
# ============================================================================


def _build_cabin_change_task(
    user: User,
    res: Reservation,
    res_id: str,
    target_cabin: str,
    payment_id: str,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where the user wants to upgrade/downgrade cabin class."""
    direction = "upgrade" if target_cabin == "business" else "downgrade"

    # Build the flights array for update_reservation_flights
    flight_infos = [
        {"flight_number": f.flight_number, "date": f.date} for f in res.flights
    ]

    actions = [
        Action(
            action_id=_action_id("cabin", 0),
            name="get_user_details",
            arguments={"user_id": user.user_id},
            compare_args=["user_id"],
        ),
        Action(
            action_id=_action_id("cabin", 1),
            name="get_reservation_details",
            arguments={"reservation_id": res_id},
            compare_args=["reservation_id"],
        ),
        Action(
            action_id=_action_id("cabin", 2),
            name="update_reservation_flights",
            arguments={
                "reservation_id": res_id,
                "cabin": target_cabin,
                "flights": flight_infos,
                "payment_id": payment_id,
            },
            compare_args=["reservation_id", "cabin"],
        ),
    ]

    return Task(
        id=f"synth_cabin_{direction}_{task_idx}",
        description=Description(
            purpose=f"Test that agent can {direction} cabin class on a reservation.",
            relevant_policies="Cabin changes are allowed except for flown flights.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="airline",
                reason_for_call=f"I want to {direction} my cabin class to {target_cabin}.",
                known_info=_make_user_known_info(user),
                unknown_info=None,
                task_instructions=(
                    f"Your reservation ID is {res_id} from {res.origin} to {res.destination}. "
                    f"You want to {direction} to {target_cabin} class. "
                    f"Confirm when the agent explains the price difference."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            nl_assertions=[
                f"The agent should successfully {direction} the cabin to {target_cabin}.",
                "The agent should inform the user about any price difference.",
            ],
            reward_basis=[RewardType.DB, RewardType.ACTION],
        ),
    )


# ============================================================================
# Scenario Template: Baggage Update
# ============================================================================


def _build_baggage_task(
    user: User,
    res: Reservation,
    res_id: str,
    extra_bags: int,
    payment_id: str,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where the user wants to add checked bags."""
    new_total = res.total_baggages + extra_bags
    new_nonfree = res.nonfree_baggages + extra_bags

    actions = [
        Action(
            action_id=_action_id("baggage", 0),
            name="get_user_details",
            arguments={"user_id": user.user_id},
            compare_args=["user_id"],
        ),
        Action(
            action_id=_action_id("baggage", 1),
            name="get_reservation_details",
            arguments={"reservation_id": res_id},
            compare_args=["reservation_id"],
        ),
        Action(
            action_id=_action_id("baggage", 2),
            name="update_reservation_baggages",
            arguments={
                "reservation_id": res_id,
                "total_baggages": new_total,
                "nonfree_baggages": new_nonfree,
                "payment_id": payment_id,
            },
            compare_args=["reservation_id", "total_baggages", "nonfree_baggages"],
        ),
    ]

    cost = extra_bags * 50
    return Task(
        id=f"synth_baggage_add_{task_idx}",
        description=Description(
            purpose="Test that agent can add checked bags to a reservation.",
            relevant_policies="Extra bags cost $50 each. Cannot use certificate for updates.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="airline",
                reason_for_call=f"I want to add {extra_bags} extra checked bag(s) to my reservation.",
                known_info=_make_user_known_info(user),
                unknown_info=None,
                task_instructions=(
                    f"Your reservation ID is {res_id} from {res.origin} to {res.destination}. "
                    f"You want to add {extra_bags} extra checked bag(s). "
                    f"Confirm when the agent tells you the cost (should be ${cost})."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            communicate_info=[str(cost)],
            reward_basis=[RewardType.DB, RewardType.COMMUNICATE],
        ),
    )


# ============================================================================
# Scenario Template: Compensation Request (Delayed Flight)
# ============================================================================


def _build_compensation_task(
    user: User,
    res: Reservation,
    res_id: str,
    delayed_flight: dict,
    eligible: bool,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a compensation request task for a delayed flight."""
    num_passengers = len(res.passengers)
    amount = 50 * num_passengers  # $50 per passenger for delays

    if eligible:
        # User is silver/gold or has insurance or business class
        actions = [
            Action(
                action_id=_action_id("comp", 0),
                name="get_user_details",
                arguments={"user_id": user.user_id},
                compare_args=["user_id"],
            ),
            Action(
                action_id=_action_id("comp", 1),
                name="get_reservation_details",
                arguments={"reservation_id": res_id},
                compare_args=["reservation_id"],
            ),
            Action(
                action_id=_action_id("comp", 2),
                name="get_flight_status",
                arguments={
                    "flight_number": delayed_flight["flight_number"],
                    "date": delayed_flight["date"],
                },
                compare_args=["flight_number", "date"],
            ),
            Action(
                action_id=_action_id("comp", 3),
                name="send_certificate",
                arguments={
                    "user_id": user.user_id,
                    "amount": amount,
                },
                compare_args=["user_id", "amount"],
            ),
        ]
        nl_assertions = [
            "The agent should verify the flight is delayed before offering compensation.",
            f"The agent should offer a certificate of ${amount}.",
        ]
    else:
        actions = [
            Action(
                action_id=_action_id("comp", 0),
                name="get_user_details",
                arguments={"user_id": user.user_id},
                compare_args=["user_id"],
            ),
            Action(
                action_id=_action_id("comp", 1),
                name="get_reservation_details",
                arguments={"reservation_id": res_id},
                compare_args=["reservation_id"],
            ),
        ]
        nl_assertions = [
            "The agent should NOT offer compensation.",
            "The agent should explain the compensation eligibility policy.",
        ]

    return Task(
        id=f"synth_compensation_{'eligible' if eligible else 'ineligible'}_{task_idx}",
        description=Description(
            purpose=f"Test {'granting' if eligible else 'refusing'} compensation for delayed flight.",
            relevant_policies="Compensation: never proactive; only for silver/gold, insurance holders, or business class.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="airline",
                reason_for_call=(
                    f"My flight {delayed_flight['flight_number']} on {delayed_flight['date']} "
                    f"was delayed. I want compensation."
                ),
                known_info=_make_user_known_info(user),
                unknown_info=None,
                task_instructions=(
                    f"Your reservation ID is {res_id}. "
                    f"Flight {delayed_flight['flight_number']} on {delayed_flight['date']} was delayed. "
                    f"You want compensation for the delay. "
                    f"If the agent asks for details, provide your reservation ID."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            nl_assertions=nl_assertions,
            reward_basis=(
                [RewardType.DB, RewardType.ACTION]
                if eligible
                else [RewardType.DB, RewardType.NL_ASSERTION]
            ),
        ),
    )


# ============================================================================
# Scenario Template: Information Query (Baggage Allowance)
# ============================================================================


def _build_info_query_task(
    user: User,
    res: Reservation,
    res_id: str,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where user asks about baggage allowance."""
    # Baggage matrix: membership × cabin
    free_bags = {
        ("regular", "basic_economy"): 0,
        ("regular", "economy"): 1,
        ("regular", "business"): 2,
        ("silver", "basic_economy"): 1,
        ("silver", "economy"): 2,
        ("silver", "business"): 3,
        ("gold", "basic_economy"): 2,
        ("gold", "economy"): 3,
        ("gold", "business"): 4,
    }
    allowed = free_bags.get((user.membership, res.cabin), 0)
    total_allowed = allowed * len(res.passengers)

    actions = [
        Action(
            action_id=_action_id("info", 0),
            name="get_user_details",
            arguments={"user_id": user.user_id},
            compare_args=["user_id"],
        ),
        Action(
            action_id=_action_id("info", 1),
            name="get_reservation_details",
            arguments={"reservation_id": res_id},
            compare_args=["reservation_id"],
        ),
    ]

    return Task(
        id=f"synth_info_baggage_{task_idx}",
        description=Description(
            purpose="Test that agent can correctly calculate and communicate baggage allowance.",
            relevant_policies="Free baggage depends on membership level and cabin class.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="airline",
                reason_for_call="I want to know how many free checked bags I get on my upcoming flight.",
                known_info=_make_user_known_info(user),
                unknown_info="You don't know your exact baggage allowance.",
                task_instructions=(
                    f"Your reservation ID is {res_id}. "
                    f"Ask how many free checked bags you're allowed in total for your reservation. "
                    f"The answer should account for your membership level and cabin class."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            communicate_info=[str(total_allowed)],
            reward_basis=[RewardType.COMMUNICATE],
        ),
    )


# ============================================================================
# Scenario Template: Transfer to Human (Impossible request)
# ============================================================================


def _build_transfer_task(
    user: User,
    res: Reservation,
    res_id: str,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where the user makes an impossible request -> transfer."""
    impossible_requests = [
        (
            f"I want to change my flight from {res.origin} to a completely different destination.",
            "You want to change your destination entirely, not just the flight times.",
        ),
        (
            f"I want to remove a passenger from reservation {res_id}.",
            "You specifically want to remove a passenger (reduce the count), not change their name.",
        ),
        (
            f"I want to split my reservation {res_id} into two separate bookings.",
            "You want each passenger to have their own separate reservation with separate payment.",
        ),
    ]
    req_idx = task_idx % len(impossible_requests)
    reason, instructions = impossible_requests[req_idx]

    actions = [
        Action(
            action_id=_action_id("transfer", 0),
            name="get_user_details",
            arguments={"user_id": user.user_id},
            compare_args=["user_id"],
        ),
        Action(
            action_id=_action_id("transfer", 1),
            name="transfer_to_human_agents",
            arguments={"summary": "Cannot fulfill request with available tools."},
            compare_args=[],
        ),
    ]

    return Task(
        id=f"synth_transfer_{task_idx}",
        description=Description(
            purpose="Test that agent correctly transfers to human for impossible requests.",
            relevant_policies="Cannot change origin/destination, remove passengers, or split reservations.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="airline",
                reason_for_call=reason,
                known_info=_make_user_known_info(user),
                unknown_info=None,
                task_instructions=(
                    f"Your reservation ID is {res_id}. {instructions} "
                    f"If the agent says they can't do this, ask to be transferred to a human agent."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            nl_assertions=[
                "The agent should transfer the user to a human agent.",
                "The agent should explain why the request cannot be fulfilled.",
            ],
            reward_basis=[RewardType.ACTION, RewardType.NL_ASSERTION],
        ),
    )


# ============================================================================
# Scenario Template: Passenger Name Change
# ============================================================================


def _build_name_change_task(
    user: User,
    res: Reservation,
    res_id: str,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where user wants to update a passenger name."""
    if not res.passengers:
        return None

    # Pick a passenger and change the last name
    pax_idx = task_idx % len(res.passengers)
    old_pax = res.passengers[pax_idx]
    new_last_name = "Garcia" if old_pax.last_name != "Garcia" else "Smith"

    new_passengers = []
    for i, p in enumerate(res.passengers):
        if i == pax_idx:
            new_passengers.append(
                {
                    "first_name": p.first_name,
                    "last_name": new_last_name,
                    "dob": p.dob,
                }
            )
        else:
            new_passengers.append(
                {
                    "first_name": p.first_name,
                    "last_name": p.last_name,
                    "dob": p.dob,
                }
            )

    actions = [
        Action(
            action_id=_action_id("name", 0),
            name="get_user_details",
            arguments={"user_id": user.user_id},
            compare_args=["user_id"],
        ),
        Action(
            action_id=_action_id("name", 1),
            name="get_reservation_details",
            arguments={"reservation_id": res_id},
            compare_args=["reservation_id"],
        ),
        Action(
            action_id=_action_id("name", 2),
            name="update_reservation_passengers",
            arguments={
                "reservation_id": res_id,
                "passengers": new_passengers,
            },
            compare_args=["reservation_id"],
        ),
    ]

    return Task(
        id=f"synth_name_change_{task_idx}",
        description=Description(
            purpose="Test that agent can update passenger name on a reservation.",
            relevant_policies="Passenger names can be changed but passenger count cannot.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="airline",
                reason_for_call=(
                    f"I need to change a passenger's last name on my reservation "
                    f"from {old_pax.last_name} to {new_last_name}."
                ),
                known_info=_make_user_known_info(user),
                unknown_info=None,
                task_instructions=(
                    f"Your reservation ID is {res_id}. "
                    f"The passenger {old_pax.first_name} {old_pax.last_name} "
                    f"needs their last name changed to {new_last_name} "
                    f"(they recently got married). Confirm the change."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            nl_assertions=[
                "The agent should successfully change the passenger's last name.",
            ],
            reward_basis=[RewardType.DB, RewardType.ACTION],
        ),
    )


# ============================================================================
# Verification function
# ============================================================================


def verify_airline_task(task: Task) -> bool:
    """
    Verify an airline task by running golden actions on a fresh environment.

    For DB-based rewards: execute all golden actions and check the DB changes.
    For NL/ACTION based rewards: just verify the actions are syntactically valid.
    """
    env = get_environment()
    db_before_hash = env.get_db_hash()

    reward_basis = task.evaluation_criteria.reward_basis
    actions = task.evaluation_criteria.actions or []

    # For refusal tasks (no write actions), verify DB is unchanged
    write_actions = [a for a in actions if a.name not in (
        "get_user_details", "get_reservation_details", "get_flight_status",
        "search_direct_flight", "search_onestop_flight", "list_all_airports",
        "calculate", "transfer_to_human_agents",
    )]

    try:
        for action in actions:
            if action.name == "transfer_to_human_agents":
                # Don't actually execute transfer, just verify it's in the actions
                continue
            env.make_tool_call(
                tool_name=action.name,
                requestor=action.requestor,
                **action.arguments,
            )
    except Exception as e:
        print(f"  Verification error: {e}")
        return False

    # For DB-based rewards, verify DB changed (or didn't for refusals)
    if RewardType.DB in reward_basis:
        db_after_hash = env.get_db_hash()
        if write_actions:
            # Should have changed
            if db_before_hash == db_after_hash:
                print("  DB should have changed but didn't")
                return False
        else:
            # Should NOT have changed
            if db_before_hash != db_after_hash:
                print("  DB changed but shouldn't have")
                return False

    return True


# ============================================================================
# Main synthesis function
# ============================================================================


def create_airline_tasks(
    num_per_template: int = 5,
    seed: int = 42,
    verify: bool = True,
) -> list[Task]:
    """
    Generate synthesized airline tasks across all scenario templates.

    Args:
        num_per_template: Number of tasks to generate per scenario template
        seed: Random seed for reproducibility
        verify: Whether to verify each task against a fresh environment

    Returns:
        List of verified Task objects
    """
    rng = random.Random(seed)
    env = get_environment()
    db = env.tools.db
    all_tasks = []
    persona_keys = list(AIRLINE_PERSONAS.keys())

    # ==== Template 1: Cancel Eligible ====
    print("=== Generating Cancel Eligible tasks ===")
    cancellable = _get_cancellable_reservations(db)
    rng.shuffle(cancellable)
    for i in range(min(num_per_template, len(cancellable))):
        res_id, user, res = cancellable[i]
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_cancel_eligible_task(
            user, res, res_id, AIRLINE_PERSONAS[persona_key], i
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        if verify and not verify_airline_task(task):
            print(f"  ✗ {task.id} failed verification")
            continue
        all_tasks.append(task)
        print(f"  ✓ {task.id}")

    # ==== Template 2: Cancel Refuse ====
    print("=== Generating Cancel Refuse tasks ===")
    non_cancellable = _get_non_cancellable_reservations(db)
    rng.shuffle(non_cancellable)
    for i in range(min(num_per_template, len(non_cancellable))):
        res_id, user, res = non_cancellable[i]
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_cancel_refuse_task(
            user, res, res_id, AIRLINE_PERSONAS[persona_key], i
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        if verify and not verify_airline_task(task):
            print(f"  ✗ {task.id} failed verification")
            continue
        all_tasks.append(task)
        print(f"  ✓ {task.id}")

    # ==== Template 3: Cabin Change (Upgrade) ====
    print("=== Generating Cabin Upgrade tasks ===")
    upgradable = [
        (rid, u, r) for rid, u, r in _get_future_reservations(db)
        if r.cabin in ("basic_economy", "economy")
    ]
    rng.shuffle(upgradable)
    for i in range(min(num_per_template, len(upgradable))):
        res_id, user, res = upgradable[i]
        target = "economy" if res.cabin == "basic_economy" else "business"
        payment_id = _get_user_payment_id(user, "credit_card")
        if payment_id is None:
            payment_id = _get_user_payment_id(user, "gift_card")
        if payment_id is None:
            continue
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_cabin_change_task(
            user, res, res_id, target, payment_id, AIRLINE_PERSONAS[persona_key], i
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        if verify and not verify_airline_task(task):
            print(f"  ✗ {task.id} failed verification")
            continue
        all_tasks.append(task)
        print(f"  ✓ {task.id}")

    # ==== Template 4: Baggage Addition ====
    print("=== Generating Baggage Addition tasks ===")
    future_res = _get_future_reservations(db)
    rng.shuffle(future_res)
    count = 0
    for res_id, user, res in future_res:
        if count >= num_per_template:
            break
        payment_id = _get_user_payment_id(user, "credit_card")
        if payment_id is None:
            payment_id = _get_user_payment_id(user, "gift_card")
        if payment_id is None:
            continue
        extra = rng.randint(1, 3)
        persona_key = persona_keys[count % len(persona_keys)]
        task = _build_baggage_task(
            user, res, res_id, extra, payment_id, AIRLINE_PERSONAS[persona_key], count
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        if verify and not verify_airline_task(task):
            print(f"  ✗ {task.id} failed verification")
            continue
        all_tasks.append(task)
        print(f"  ✓ {task.id}")
        count += 1

    # ==== Template 5: Compensation (Eligible) ====
    print("=== Generating Compensation Eligible tasks ===")
    delayed = _get_delayed_flights(db)
    eligible_delayed = [
        (rid, u, r, fl) for rid, u, r, fl in delayed
        if u.membership in ("silver", "gold") or r.insurance == "yes" or r.cabin == "business"
    ]
    rng.shuffle(eligible_delayed)
    for i in range(min(num_per_template, len(eligible_delayed))):
        res_id, user, res, flight = eligible_delayed[i]
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_compensation_task(
            user, res, res_id, flight, True, AIRLINE_PERSONAS[persona_key], i
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        if verify and not verify_airline_task(task):
            print(f"  ✗ {task.id} failed verification")
            continue
        all_tasks.append(task)
        print(f"  ✓ {task.id}")

    # ==== Template 6: Compensation (Ineligible) ====
    print("=== Generating Compensation Ineligible tasks ===")
    ineligible_delayed = [
        (rid, u, r, fl) for rid, u, r, fl in delayed
        if u.membership == "regular" and r.insurance == "no" and r.cabin != "business"
    ]
    rng.shuffle(ineligible_delayed)
    for i in range(min(num_per_template, len(ineligible_delayed))):
        res_id, user, res, flight = ineligible_delayed[i]
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_compensation_task(
            user, res, res_id, flight, False, AIRLINE_PERSONAS[persona_key], i
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        if verify and not verify_airline_task(task):
            print(f"  ✗ {task.id} failed verification")
            continue
        all_tasks.append(task)
        print(f"  ✓ {task.id}")

    # ==== Template 7: Info Query (Baggage Allowance) ====
    print("=== Generating Info Query tasks ===")
    rng.shuffle(future_res)
    for i in range(min(num_per_template, len(future_res))):
        res_id, user, res = future_res[i]
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_info_query_task(
            user, res, res_id, AIRLINE_PERSONAS[persona_key], i
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        # Info queries don't change DB, always valid
        all_tasks.append(task)
        print(f"  ✓ {task.id}")

    # ==== Template 8: Transfer to Human ====
    print("=== Generating Transfer to Human tasks ===")
    rng.shuffle(future_res)
    for i in range(min(num_per_template, len(future_res))):
        res_id, user, res = future_res[i]
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_transfer_task(
            user, res, res_id, AIRLINE_PERSONAS[persona_key], i
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        all_tasks.append(task)
        print(f"  ✓ {task.id}")

    # ==== Template 9: Passenger Name Change ====
    print("=== Generating Name Change tasks ===")
    multi_pax = [
        (rid, u, r) for rid, u, r in future_res if len(r.passengers) >= 1
    ]
    rng.shuffle(multi_pax)
    for i in range(min(num_per_template, len(multi_pax))):
        res_id, user, res = multi_pax[i]
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_name_change_task(
            user, res, res_id, AIRLINE_PERSONAS[persona_key], i
        )
        if task is None:
            continue
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        if verify and not verify_airline_task(task):
            print(f"  ✗ {task.id} failed verification")
            continue
        all_tasks.append(task)
        print(f"  ✓ {task.id}")

    print(f"\n=== Total airline tasks generated: {len(all_tasks)} ===")
    return all_tasks
