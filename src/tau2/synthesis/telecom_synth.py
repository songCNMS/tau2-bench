"""
Telecom domain task synthesizer (account management).

Generates diverse telecom customer service tasks by:
1. Sampling real customers, lines, bills, and plans from the database
2. Applying scenario templates that test different account-management aspects
3. Verifying each task by executing golden actions against a fresh environment

Task categories:
- Billing inquiry (view bills)
- Payment request (send payment for overdue/issued bills)
- Line suspension
- Line resumption (eligible + ineligible)
- Enable / disable roaming
- Data refueling
- Data usage inquiry
- Account info query
- Transfer to human (impossible requests)

These tasks complement the existing troubleshooting tasks in
tau2.domains.telecom.tasks, which focus on device-level issues.
Each template is parameterized over the database, making synthesis scalable:
swap in any customer/line and get a valid task.
"""

import random
from typing import Optional

from tau2.data_model.tasks import (
    Action,
    Description,
    EvaluationCriteria,
    RewardType,
    StructuredUserInstructions,
    Task,
    UserScenario,
)
from tau2.domains.telecom.data_model import (
    AccountStatus,
    BillStatus,
    Customer,
    Line,
    LineStatus,
    TelecomDB,
)
from tau2.domains.telecom.environment import TelecomEnvironment, get_environment


# ============================================================================
# Persona definitions (4 + None, telecom-specific)
# ============================================================================

TELECOM_PERSONAS = {
    "None": None,
    "Polite": (
        "You are a polite, patient customer who provides information when asked. "
        "You follow instructions carefully and thank the agent for their help."
    ),
    "Frustrated": (
        "You are a frustrated customer who has been having recurring issues with "
        "your telecom service. You are short-tempered and may express dissatisfaction, "
        "but you cooperate when the agent is helpful. You want things resolved quickly."
    ),
    "Confused": (
        "You are an older customer who is not very tech-savvy. You sometimes mix up "
        "telecom terminology (e.g., confuse 'line' with 'phone number', or 'plan' "
        "with 'bill'). You need things explained simply and may ask for clarification."
    ),
    "Busy": (
        "You are a busy professional who wants things done fast. You give terse, "
        "minimal answers and get impatient if asked too many questions. You may "
        "provide multiple pieces of information at once to speed things up."
    ),
}


# ============================================================================
# Helper functions for querying the database
# ============================================================================


def _get_active_customers(db: TelecomDB) -> list[Customer]:
    """Get customers with Active account status."""
    return [c for c in db.customers if c.account_status == AccountStatus.ACTIVE]


def _get_customer_active_lines(db: TelecomDB, customer: Customer) -> list[Line]:
    """Get active lines for a customer."""
    results = []
    for line_id in customer.line_ids:
        line = _get_line_by_id(db, line_id)
        if line and line.status == LineStatus.ACTIVE:
            results.append(line)
    return results


def _get_customer_suspended_lines(db: TelecomDB, customer: Customer) -> list[Line]:
    """Get suspended lines for a customer."""
    results = []
    for line_id in customer.line_ids:
        line = _get_line_by_id(db, line_id)
        if line and line.status == LineStatus.SUSPENDED:
            results.append(line)
    return results


def _get_lines_with_roaming(
    db: TelecomDB, enabled: bool
) -> list[tuple[Customer, Line]]:
    """Get (customer, line) pairs where roaming matches the desired state."""
    results = []
    for customer in _get_active_customers(db):
        for line_id in customer.line_ids:
            line = _get_line_by_id(db, line_id)
            if (
                line
                and line.status == LineStatus.ACTIVE
                and line.roaming_enabled == enabled
            ):
                results.append((customer, line))
    return results


def _get_overdue_bills(db: TelecomDB) -> list[tuple[Customer, "Bill"]]:
    """Get (customer, bill) pairs for overdue bills."""
    from tau2.domains.telecom.data_model import Bill

    results = []
    for customer in db.customers:
        for bill_id in customer.bill_ids:
            bill = _get_bill_by_id(db, bill_id)
            if bill and bill.status == BillStatus.OVERDUE:
                results.append((customer, bill))
    return results


def _get_issued_bills(db: TelecomDB) -> list[tuple[Customer, "Bill"]]:
    """Get (customer, bill) pairs for issued bills."""
    from tau2.domains.telecom.data_model import Bill

    results = []
    for customer in db.customers:
        for bill_id in customer.bill_ids:
            bill = _get_bill_by_id(db, bill_id)
            if bill and bill.status == BillStatus.ISSUED:
                results.append((customer, bill))
    return results


def _get_plan_by_id(db: TelecomDB, plan_id: str):
    """Look up a plan by ID."""
    for plan in db.plans:
        if plan.plan_id == plan_id:
            return plan
    return None


def _get_line_by_id(db: TelecomDB, line_id: str) -> Optional[Line]:
    """Look up a line by ID."""
    for line in db.lines:
        if line.line_id == line_id:
            return line
    return None


def _get_bill_by_id(db: TelecomDB, bill_id: str):
    """Look up a bill by ID."""
    for bill in db.bills:
        if bill.bill_id == bill_id:
            return bill
    return None


def _action_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx}"


# ============================================================================
# Three authentication methods — cycled per template
# ============================================================================
# Method 0: phone number  -> get_customer_by_phone
# Method 1: name + DOB    -> get_customer_by_name
# Method 2: customer ID   -> get_customer_by_id


def _get_known_info(customer: Customer, task_idx: int) -> str:
    """Return the known_info string for the chosen auth method."""
    method = task_idx % 3
    if method == 0:
        return f"Your phone number is {customer.phone_number}."
    elif method == 1:
        return (
            f"Your name is {customer.full_name}. "
            f"Your date of birth is {customer.date_of_birth}."
        )
    else:
        return f"Your customer ID is {customer.customer_id}."


def _get_lookup_action(
    customer: Customer, task_idx: int, prefix: str, idx: int
) -> Action:
    """Generate the authentication/lookup Action matching _get_known_info."""
    method = task_idx % 3
    if method == 0:
        return Action(
            action_id=_action_id(prefix, idx),
            name="get_customer_by_phone",
            arguments={"phone_number": customer.phone_number},
            compare_args=["phone_number"],
        )
    elif method == 1:
        return Action(
            action_id=_action_id(prefix, idx),
            name="get_customer_by_name",
            arguments={
                "full_name": customer.full_name,
                "dob": customer.date_of_birth,
            },
            compare_args=["full_name", "dob"],
        )
    else:
        return Action(
            action_id=_action_id(prefix, idx),
            name="get_customer_by_id",
            arguments={"customer_id": customer.customer_id},
            compare_args=["customer_id"],
        )


# ============================================================================
# Template 1: Billing Inquiry (READ)
# ============================================================================


def _build_billing_inquiry_task(
    customer: Customer,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where the user wants to view their recent bills."""
    actions = [
        _get_lookup_action(customer, task_idx, "billing_inq", 0),
        Action(
            action_id=_action_id("billing_inq", 1),
            name="get_bills_for_customer",
            arguments={"customer_id": customer.customer_id},
            compare_args=["customer_id"],
        ),
    ]

    return Task(
        id=f"synth_billing_inquiry_{task_idx}",
        description=Description(
            purpose="Test that agent can look up and communicate billing information.",
            relevant_policies="Agent should verify customer identity before accessing billing.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="telecom",
                reason_for_call="I want to see my recent bills and know what I owe.",
                known_info=_get_known_info(customer, task_idx),
                unknown_info="You don't know the exact bill amounts or statuses.",
                task_instructions=(
                    "Ask the agent to look up your recent bills. "
                    "You want to know the total amounts due and the status of each bill."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            communicate_info=["bills", "total"],
            reward_basis=[RewardType.COMMUNICATE],
        ),
    )


# ============================================================================
# Template 2: Payment Request (WRITE)
# ============================================================================


def _build_payment_request_task(
    customer: Customer,
    bill,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where the user wants to pay an outstanding bill."""
    actions = [
        _get_lookup_action(customer, task_idx, "payment", 0),
        Action(
            action_id=_action_id("payment", 1),
            name="get_bills_for_customer",
            arguments={"customer_id": customer.customer_id},
            compare_args=["customer_id"],
        ),
        Action(
            action_id=_action_id("payment", 2),
            name="send_payment_request",
            arguments={
                "customer_id": customer.customer_id,
                "bill_id": bill.bill_id,
            },
            compare_args=["customer_id", "bill_id"],
        ),
    ]

    return Task(
        id=f"synth_payment_request_{task_idx}",
        description=Description(
            purpose="Test that agent can send a payment request for an outstanding bill.",
            relevant_policies=(
                "Agent must identify the correct bill and send a payment request. "
                "Only one bill can be awaiting payment at a time."
            ),
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="telecom",
                reason_for_call=(
                    f"I want to pay my bill. I believe I owe ${bill.total_due:.2f}."
                ),
                known_info=_get_known_info(customer, task_idx),
                unknown_info=None,
                task_instructions=(
                    f"You want to pay bill {bill.bill_id} (${bill.total_due:.2f}). "
                    "Ask the agent to send you a payment request so you can pay it. "
                    "Confirm when the agent identifies the correct bill."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            nl_assertions=[
                "The agent should identify the correct outstanding bill.",
                "The agent should send a payment request to the customer.",
            ],
            reward_basis=[RewardType.DB, RewardType.ACTION],
        ),
    )


# ============================================================================
# Template 3: Line Suspension (WRITE)
# ============================================================================


def _build_line_suspension_task(
    customer: Customer,
    line: Line,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where the user wants to suspend an active line."""
    reasons = [
        "I'm traveling abroad and want to suspend this line temporarily.",
        "This line belongs to my child who lost their phone.",
        "I want to reduce costs temporarily by suspending this line.",
    ]
    reason = reasons[task_idx % len(reasons)]

    actions = [
        _get_lookup_action(customer, task_idx, "suspend", 0),
        Action(
            action_id=_action_id("suspend", 1),
            name="get_details_by_id",
            arguments={"id": line.line_id},
            compare_args=["id"],
        ),
        Action(
            action_id=_action_id("suspend", 2),
            name="suspend_line",
            arguments={
                "customer_id": customer.customer_id,
                "line_id": line.line_id,
                "reason": reason,
            },
            compare_args=["customer_id", "line_id"],
        ),
    ]

    return Task(
        id=f"synth_line_suspend_{task_idx}",
        description=Description(
            purpose="Test that agent can suspend an active line per customer request.",
            relevant_policies="Only active lines can be suspended. A $5/month holding fee applies.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="telecom",
                reason_for_call=reason,
                known_info=_get_known_info(customer, task_idx),
                unknown_info=None,
                task_instructions=(
                    f"You want to suspend line {line.line_id} "
                    f"(phone number {line.phone_number}). "
                    "Confirm when the agent explains the suspension details and fees."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            nl_assertions=[
                "The agent should successfully suspend the line.",
                "The agent should inform the customer about the holding fee.",
            ],
            reward_basis=[RewardType.DB, RewardType.ACTION],
        ),
    )


# ============================================================================
# Template 4: Line Resumption — Eligible (WRITE)
# ============================================================================


def _build_line_resume_eligible_task(
    customer: Customer,
    line: Line,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where the user wants to resume a suspended line (eligible)."""
    actions = [
        _get_lookup_action(customer, task_idx, "resume", 0),
        Action(
            action_id=_action_id("resume", 1),
            name="get_details_by_id",
            arguments={"id": line.line_id},
            compare_args=["id"],
        ),
        Action(
            action_id=_action_id("resume", 2),
            name="resume_line",
            arguments={
                "customer_id": customer.customer_id,
                "line_id": line.line_id,
            },
            compare_args=["customer_id", "line_id"],
        ),
    ]

    return Task(
        id=f"synth_line_resume_eligible_{task_idx}",
        description=Description(
            purpose="Test that agent can resume a suspended line.",
            relevant_policies="Suspended or pending activation lines can be resumed.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="telecom",
                reason_for_call=(
                    f"I want to reactivate my suspended line {line.line_id}."
                ),
                known_info=_get_known_info(customer, task_idx),
                unknown_info=None,
                task_instructions=(
                    f"You want to resume line {line.line_id} "
                    f"(phone number {line.phone_number}). "
                    "Confirm the reactivation when the agent is ready."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            nl_assertions=[
                "The agent should successfully resume the line.",
            ],
            reward_basis=[RewardType.DB, RewardType.ACTION],
        ),
    )


# ============================================================================
# Template 5: Line Resumption — Ineligible (REFUSE)
# ============================================================================


def _build_line_resume_ineligible_task(
    customer: Customer,
    line: Line,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where the user wants to resume a line that is already active or closed."""
    actions = [
        _get_lookup_action(customer, task_idx, "resume_inelig", 0),
        Action(
            action_id=_action_id("resume_inelig", 1),
            name="get_details_by_id",
            arguments={"id": line.line_id},
            compare_args=["id"],
        ),
    ]

    return Task(
        id=f"synth_line_resume_ineligible_{task_idx}",
        description=Description(
            purpose="Test that agent correctly refuses to resume a line that is not suspended.",
            relevant_policies="Only suspended or pending activation lines can be resumed.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="telecom",
                reason_for_call=(
                    f"I want to reactivate line {line.line_id}. "
                    "I think it might have been suspended."
                ),
                known_info=_get_known_info(customer, task_idx),
                unknown_info="You are not sure of the current status of this line.",
                task_instructions=(
                    f"You believe line {line.line_id} was suspended and want to resume it. "
                    "If the agent says it's already active, accept that."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            nl_assertions=[
                "The agent should NOT resume the line (it is already active).",
                "The agent should explain the current status of the line.",
            ],
            reward_basis=[RewardType.DB, RewardType.NL_ASSERTION],
        ),
    )


# ============================================================================
# Template 6: Enable Roaming (WRITE)
# ============================================================================


def _build_enable_roaming_task(
    customer: Customer,
    line: Line,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where the user wants to enable international roaming."""
    actions = [
        _get_lookup_action(customer, task_idx, "roam_on", 0),
        Action(
            action_id=_action_id("roam_on", 1),
            name="get_details_by_id",
            arguments={"id": line.line_id},
            compare_args=["id"],
        ),
        Action(
            action_id=_action_id("roam_on", 2),
            name="enable_roaming",
            arguments={
                "customer_id": customer.customer_id,
                "line_id": line.line_id,
            },
            compare_args=["customer_id", "line_id"],
        ),
    ]

    return Task(
        id=f"synth_enable_roaming_{task_idx}",
        description=Description(
            purpose="Test that agent can enable international roaming on a line.",
            relevant_policies="Roaming can be enabled on active lines.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="telecom",
                reason_for_call=(
                    "I'm traveling internationally and need to enable roaming "
                    f"on my line {line.line_id}."
                ),
                known_info=_get_known_info(customer, task_idx),
                unknown_info=None,
                task_instructions=(
                    f"You want to enable international roaming on line {line.line_id} "
                    f"(phone number {line.phone_number}). "
                    "Confirm when the agent is ready to enable it."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            nl_assertions=[
                "The agent should successfully enable roaming.",
            ],
            reward_basis=[RewardType.DB, RewardType.ACTION],
        ),
    )


# ============================================================================
# Template 7: Disable Roaming (WRITE)
# ============================================================================


def _build_disable_roaming_task(
    customer: Customer,
    line: Line,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where the user wants to disable international roaming."""
    actions = [
        _get_lookup_action(customer, task_idx, "roam_off", 0),
        Action(
            action_id=_action_id("roam_off", 1),
            name="get_details_by_id",
            arguments={"id": line.line_id},
            compare_args=["id"],
        ),
        Action(
            action_id=_action_id("roam_off", 2),
            name="disable_roaming",
            arguments={
                "customer_id": customer.customer_id,
                "line_id": line.line_id,
            },
            compare_args=["customer_id", "line_id"],
        ),
    ]

    return Task(
        id=f"synth_disable_roaming_{task_idx}",
        description=Description(
            purpose="Test that agent can disable international roaming on a line.",
            relevant_policies="Roaming can be disabled on lines where it is currently enabled.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="telecom",
                reason_for_call=(
                    "I'm back from my trip and want to turn off roaming "
                    f"on line {line.line_id}."
                ),
                known_info=_get_known_info(customer, task_idx),
                unknown_info=None,
                task_instructions=(
                    f"You want to disable international roaming on line {line.line_id} "
                    f"(phone number {line.phone_number}). "
                    "Confirm when the agent is ready to disable it."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            nl_assertions=[
                "The agent should successfully disable roaming.",
            ],
            reward_basis=[RewardType.DB, RewardType.ACTION],
        ),
    )


# ============================================================================
# Template 8: Data Refueling (WRITE)
# ============================================================================


def _build_data_refuel_task(
    customer: Customer,
    line: Line,
    db: TelecomDB,
    persona: Optional[str],
    task_idx: int,
) -> Optional[Task]:
    """Build a task where the user wants to add data to their line."""
    plan = _get_plan_by_id(db, line.plan_id)
    if plan is None:
        return None

    gb_amount = 2.0  # Standard refuel amount
    cost = gb_amount * plan.data_refueling_price_per_gb

    actions = [
        _get_lookup_action(customer, task_idx, "refuel", 0),
        Action(
            action_id=_action_id("refuel", 1),
            name="get_data_usage",
            arguments={
                "customer_id": customer.customer_id,
                "line_id": line.line_id,
            },
            compare_args=["customer_id", "line_id"],
        ),
        Action(
            action_id=_action_id("refuel", 2),
            name="refuel_data",
            arguments={
                "customer_id": customer.customer_id,
                "line_id": line.line_id,
                "gb_amount": gb_amount,
            },
            compare_args=["customer_id", "line_id", "gb_amount"],
        ),
    ]

    return Task(
        id=f"synth_data_refuel_{task_idx}",
        description=Description(
            purpose="Test that agent can refuel data for a customer's line.",
            relevant_policies=(
                "Data refueling charges at the plan's per-GB rate. "
                "Agent should check current usage before refueling."
            ),
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="telecom",
                reason_for_call=(
                    f"I'm running low on data on line {line.line_id} and want to add more."
                ),
                known_info=_get_known_info(customer, task_idx),
                unknown_info="You don't know the exact refueling cost per GB.",
                task_instructions=(
                    f"You want to add {gb_amount} GB of data to line {line.line_id}. "
                    f"The cost should be ${cost:.2f} based on your plan's rate of "
                    f"${plan.data_refueling_price_per_gb}/GB. "
                    "Confirm the refueling when the agent tells you the cost."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            communicate_info=[f"{cost:.2f}"],
            reward_basis=[RewardType.DB, RewardType.COMMUNICATE],
        ),
    )


# ============================================================================
# Template 9: Data Usage Inquiry (READ)
# ============================================================================


def _build_data_usage_inquiry_task(
    customer: Customer,
    line: Line,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where the user asks about their data usage."""
    actions = [
        _get_lookup_action(customer, task_idx, "data_usage", 0),
        Action(
            action_id=_action_id("data_usage", 1),
            name="get_data_usage",
            arguments={
                "customer_id": customer.customer_id,
                "line_id": line.line_id,
            },
            compare_args=["customer_id", "line_id"],
        ),
    ]

    return Task(
        id=f"synth_data_usage_inquiry_{task_idx}",
        description=Description(
            purpose="Test that agent can look up and communicate data usage information.",
            relevant_policies="Agent should verify identity before disclosing usage.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="telecom",
                reason_for_call=(
                    f"I want to check how much data I've used on line {line.line_id}."
                ),
                known_info=_get_known_info(customer, task_idx),
                unknown_info="You don't know your current data usage or limit.",
                task_instructions=(
                    f"Ask the agent about your data usage on line {line.line_id}. "
                    "You want to know how much data you've used, your limit, "
                    "and when the billing cycle ends."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            communicate_info=["data_used", "data_limit"],
            reward_basis=[RewardType.COMMUNICATE],
        ),
    )


# ============================================================================
# Template 10: Account Info Query (READ)
# ============================================================================


def _build_account_info_task(
    customer: Customer,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where the user asks about their account details."""
    # Optionally also look up a line detail
    has_lines = len(customer.line_ids) > 0
    actions = [
        _get_lookup_action(customer, task_idx, "acct_info", 0),
    ]

    if has_lines:
        line_id = customer.line_ids[task_idx % len(customer.line_ids)]
        actions.append(
            Action(
                action_id=_action_id("acct_info", 1),
                name="get_details_by_id",
                arguments={"id": line_id},
                compare_args=["id"],
            ),
        )

    return Task(
        id=f"synth_account_info_{task_idx}",
        description=Description(
            purpose="Test that agent can communicate account and line details.",
            relevant_policies="Agent should verify identity before sharing account details.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="telecom",
                reason_for_call="I want to check on my account details and line information.",
                known_info=_get_known_info(customer, task_idx),
                unknown_info="You don't know your exact plan details or line statuses.",
                task_instructions=(
                    "Ask the agent for your account information — "
                    "your account status, what lines you have, and their current status. "
                    "If you have lines, ask about the plan on one of them."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            communicate_info=["account_status", "lines"],
            reward_basis=[RewardType.COMMUNICATE],
        ),
    )


# ============================================================================
# Template 11: Transfer to Human (REFUSE)
# ============================================================================


def _build_transfer_task(
    customer: Customer,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where the user makes an impossible request -> transfer."""
    impossible_requests = [
        (
            "I want to change my phone number to a specific custom number.",
            "You want a specific vanity number (e.g., 555-000-1234). "
            "If the agent says they can't do this, ask to be transferred.",
        ),
        (
            "I want to transfer my line to a different carrier but keep my account.",
            "You want to port your number out but keep your telecom account active "
            "with the same billing. If the agent can't help, ask for a human agent.",
        ),
        (
            "I want to dispute all my historical bills from the past year.",
            "You want a retroactive dispute of all bills. If the agent says "
            "they can't process this, ask to speak to a supervisor or human agent.",
        ),
    ]

    req_idx = task_idx % len(impossible_requests)
    reason, instructions = impossible_requests[req_idx]

    actions = [
        _get_lookup_action(customer, task_idx, "transfer", 0),
        Action(
            action_id=_action_id("transfer", 1),
            name="transfer_to_human_agents",
            arguments={"summary": "Cannot fulfill request with available tools."},
            compare_args=[],
        ),
    ]

    return Task(
        id=f"synth_telecom_transfer_{task_idx}",
        description=Description(
            purpose="Test that agent correctly transfers to human for impossible requests.",
            relevant_policies="Agent should transfer when request cannot be fulfilled with available tools.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="telecom",
                reason_for_call=reason,
                known_info=_get_known_info(customer, task_idx),
                unknown_info=None,
                task_instructions=(
                    f"{instructions}"
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            nl_assertions=[
                "The agent should explain why the request cannot be fulfilled.",
                "The agent should transfer the user to a human agent.",
            ],
            reward_basis=[RewardType.ACTION, RewardType.NL_ASSERTION],
        ),
    )


# ============================================================================
# Verification function
# ============================================================================


def verify_telecom_task(task: Task) -> bool:
    """
    Verify a telecom task by running golden actions on a fresh environment.

    Creates a fresh TelecomEnvironment, executes all golden actions, and
    checks that the DB hash changed (or didn't) as expected.
    """
    env = get_environment()
    db_before_hash = env.get_db_hash()
    actions = task.evaluation_criteria.actions or []

    # Read-only tools that do not modify the DB
    read_only_tools = {
        "get_customer_by_phone",
        "get_customer_by_id",
        "get_customer_by_name",
        "get_details_by_id",
        "get_bills_for_customer",
        "get_data_usage",
        "transfer_to_human_agents",
    }

    write_actions = [a for a in actions if a.name not in read_only_tools]

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
    reward_basis = task.evaluation_criteria.reward_basis
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


def create_telecom_tasks(
    num_per_template: int = 5,
    seed: int = 42,
    verify: bool = True,
) -> list[Task]:
    """
    Generate synthesized telecom account-management tasks across all templates.

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
    persona_keys = list(TELECOM_PERSONAS.keys())

    # ==== Template 1: Billing Inquiry ====
    print("=== Generating Billing Inquiry tasks ===")
    active_customers = _get_active_customers(db)
    rng.shuffle(active_customers)
    for i in range(min(num_per_template, len(active_customers))):
        customer = active_customers[i]
        if not customer.bill_ids:
            continue
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_billing_inquiry_task(
            customer, TELECOM_PERSONAS[persona_key], i
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        # Read-only — always valid, skip verification
        all_tasks.append(task)
        print(f"  \u2713 {task.id}")

    # ==== Template 2: Payment Request ====
    print("=== Generating Payment Request tasks ===")
    # Collect overdue + issued bills (payable)
    overdue = _get_overdue_bills(db)
    issued = _get_issued_bills(db)
    payable = overdue + issued
    rng.shuffle(payable)
    for i in range(min(num_per_template, len(payable))):
        customer, bill = payable[i]
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_payment_request_task(
            customer, bill, TELECOM_PERSONAS[persona_key], i
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        if verify and not verify_telecom_task(task):
            print(f"  \u2717 {task.id} failed verification")
            continue
        all_tasks.append(task)
        print(f"  \u2713 {task.id}")

    # ==== Template 3: Line Suspension ====
    print("=== Generating Line Suspension tasks ===")
    suspend_candidates = []
    for customer in _get_active_customers(db):
        for line in _get_customer_active_lines(db, customer):
            suspend_candidates.append((customer, line))
    rng.shuffle(suspend_candidates)
    for i in range(min(num_per_template, len(suspend_candidates))):
        customer, line = suspend_candidates[i]
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_line_suspension_task(
            customer, line, TELECOM_PERSONAS[persona_key], i
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        if verify and not verify_telecom_task(task):
            print(f"  \u2717 {task.id} failed verification")
            continue
        all_tasks.append(task)
        print(f"  \u2713 {task.id}")

    # ==== Template 4: Line Resumption (Eligible) ====
    print("=== Generating Line Resumption (Eligible) tasks ===")
    resume_candidates = []
    for customer in _get_active_customers(db):
        for line in _get_customer_suspended_lines(db, customer):
            resume_candidates.append((customer, line))
    # Also check suspended-account customers who have suspended lines
    for customer in db.customers:
        if customer.account_status == AccountStatus.SUSPENDED:
            for line in _get_customer_suspended_lines(db, customer):
                resume_candidates.append((customer, line))
    rng.shuffle(resume_candidates)
    for i in range(min(num_per_template, len(resume_candidates))):
        customer, line = resume_candidates[i]
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_line_resume_eligible_task(
            customer, line, TELECOM_PERSONAS[persona_key], i
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        if verify and not verify_telecom_task(task):
            print(f"  \u2717 {task.id} failed verification")
            continue
        all_tasks.append(task)
        print(f"  \u2713 {task.id}")

    # ==== Template 5: Line Resumption (Ineligible — already active) ====
    print("=== Generating Line Resumption (Ineligible) tasks ===")
    ineligible_resume = []
    for customer in _get_active_customers(db):
        for line in _get_customer_active_lines(db, customer):
            ineligible_resume.append((customer, line))
    rng.shuffle(ineligible_resume)
    for i in range(min(num_per_template, len(ineligible_resume))):
        customer, line = ineligible_resume[i]
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_line_resume_ineligible_task(
            customer, line, TELECOM_PERSONAS[persona_key], i
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        if verify and not verify_telecom_task(task):
            print(f"  \u2717 {task.id} failed verification")
            continue
        all_tasks.append(task)
        print(f"  \u2713 {task.id}")

    # ==== Template 6: Enable Roaming ====
    print("=== Generating Enable Roaming tasks ===")
    roaming_off = _get_lines_with_roaming(db, enabled=False)
    rng.shuffle(roaming_off)
    for i in range(min(num_per_template, len(roaming_off))):
        customer, line = roaming_off[i]
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_enable_roaming_task(
            customer, line, TELECOM_PERSONAS[persona_key], i
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        if verify and not verify_telecom_task(task):
            print(f"  \u2717 {task.id} failed verification")
            continue
        all_tasks.append(task)
        print(f"  \u2713 {task.id}")

    # ==== Template 7: Disable Roaming ====
    print("=== Generating Disable Roaming tasks ===")
    roaming_on = _get_lines_with_roaming(db, enabled=True)
    rng.shuffle(roaming_on)
    for i in range(min(num_per_template, len(roaming_on))):
        customer, line = roaming_on[i]
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_disable_roaming_task(
            customer, line, TELECOM_PERSONAS[persona_key], i
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        if verify and not verify_telecom_task(task):
            print(f"  \u2717 {task.id} failed verification")
            continue
        all_tasks.append(task)
        print(f"  \u2713 {task.id}")

    # ==== Template 8: Data Refueling ====
    print("=== Generating Data Refueling tasks ===")
    refuel_candidates = []
    for customer in _get_active_customers(db):
        for line in _get_customer_active_lines(db, customer):
            refuel_candidates.append((customer, line))
    rng.shuffle(refuel_candidates)
    count = 0
    for customer, line in refuel_candidates:
        if count >= num_per_template:
            break
        persona_key = persona_keys[count % len(persona_keys)]
        task = _build_data_refuel_task(
            customer, line, db, TELECOM_PERSONAS[persona_key], count
        )
        if task is None:
            continue
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        if verify and not verify_telecom_task(task):
            print(f"  \u2717 {task.id} failed verification")
            continue
        all_tasks.append(task)
        print(f"  \u2713 {task.id}")
        count += 1

    # ==== Template 9: Data Usage Inquiry ====
    print("=== Generating Data Usage Inquiry tasks ===")
    usage_candidates = []
    for customer in _get_active_customers(db):
        for line in _get_customer_active_lines(db, customer):
            usage_candidates.append((customer, line))
    rng.shuffle(usage_candidates)
    for i in range(min(num_per_template, len(usage_candidates))):
        customer, line = usage_candidates[i]
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_data_usage_inquiry_task(
            customer, line, TELECOM_PERSONAS[persona_key], i
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        # Read-only — always valid
        all_tasks.append(task)
        print(f"  \u2713 {task.id}")

    # ==== Template 10: Account Info Query ====
    print("=== Generating Account Info Query tasks ===")
    rng.shuffle(active_customers)
    for i in range(min(num_per_template, len(active_customers))):
        customer = active_customers[i]
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_account_info_task(
            customer, TELECOM_PERSONAS[persona_key], i
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        # Read-only — always valid
        all_tasks.append(task)
        print(f"  \u2713 {task.id}")

    # ==== Template 11: Transfer to Human ====
    print("=== Generating Transfer to Human tasks ===")
    rng.shuffle(active_customers)
    for i in range(min(num_per_template, len(active_customers))):
        customer = active_customers[i]
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_transfer_task(
            customer, TELECOM_PERSONAS[persona_key], i
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        # Transfer tasks are always valid
        all_tasks.append(task)
        print(f"  \u2713 {task.id}")

    print(f"\n=== Total telecom tasks generated: {len(all_tasks)} ===")
    return all_tasks
