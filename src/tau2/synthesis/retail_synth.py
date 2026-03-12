"""
Retail domain task synthesizer.

Generates diverse retail customer service tasks by:
1. Sampling real users, orders, and products from the database
2. Applying scenario templates that test different policy aspects
3. Verifying each task by executing golden actions against a fresh environment

Task categories:
- Cancel pending order
- Exchange delivered items
- Return delivered items
- Modify pending order items
- Modify pending order payment
- Modify pending order address
- Information queries (product variants)
- Transfer to human (impossible requests)
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
from tau2.domains.retail.data_model import (
    RetailDB,
    Order,
    User,
    Product,
    Variant,
    GiftCard,
)
from tau2.domains.retail.environment import get_environment
from tau2.environment.environment import Environment


# ============================================================================
# Persona definitions
# ============================================================================

RETAIL_PERSONAS = {
    "None": None,
    "Friendly": (
        "You are a friendly, chatty customer who likes to make small talk. "
        "You answer questions directly but sometimes add extra context about "
        "why you're making the request."
    ),
    "Impatient": (
        "You are a busy professional who wants things done quickly. "
        "You give minimal information and get frustrated if asked too many questions. "
        "You might provide information out of order or skip steps."
    ),
    "Cautious": (
        "You are a privacy-conscious customer who is reluctant to share personal "
        "information. You only provide what's absolutely necessary and question "
        "why the agent needs each piece of information."
    ),
}


# ============================================================================
# Helper functions
# ============================================================================


def _get_pending_orders(db: RetailDB) -> list[tuple[str, User, Order]]:
    """Get orders with 'pending' status."""
    results = []
    for user_id, user in db.users.items():
        for order_id in user.orders:
            if order_id not in db.orders:
                continue
            order = db.orders[order_id]
            if order.status == "pending":
                results.append((order_id, user, order))
    return results


def _get_delivered_orders(db: RetailDB) -> list[tuple[str, User, Order]]:
    """Get orders with 'delivered' status."""
    results = []
    for user_id, user in db.users.items():
        for order_id in user.orders:
            if order_id not in db.orders:
                continue
            order = db.orders[order_id]
            if order.status == "delivered":
                results.append((order_id, user, order))
    return results


def _find_alternative_variant(
    db: RetailDB, product_id: str, current_item_id: str
) -> Optional[Variant]:
    """Find an available alternative variant for a product, different from the current one."""
    product = db.products.get(product_id)
    if not product:
        return None
    for var_id, var in product.variants.items():
        if var_id != current_item_id and var.available:
            return var
    return None


def _get_user_payment_id(user: User, source_type: str = "credit_card") -> Optional[str]:
    """Get a payment method ID of a specific type from a user."""
    for pm_id, pm in user.payment_methods.items():
        if pm.source == source_type:
            return pm_id
    return None


def _get_any_payment_id(user: User) -> Optional[str]:
    """Get any valid payment method from user, preferring credit cards."""
    for source in ("credit_card", "paypal", "gift_card"):
        pm_id = _get_user_payment_id(user, source)
        if pm_id is not None:
            return pm_id
    return None


def _make_user_known_info_email(user: User) -> str:
    """Known info using email for authentication."""
    return f"Your email is {user.email}."


def _make_user_known_info_name_zip(user: User) -> str:
    """Known info using name + zip for authentication."""
    return (
        f"Your name is {user.name.first_name} {user.name.last_name}. "
        f"Your zip code is {user.address.zip}."
    )


def _action_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx}"


def _describe_item_options(options: dict[str, str]) -> str:
    """Describe item options in natural language."""
    parts = [f"{k}: {v}" for k, v in options.items()]
    return ", ".join(parts)


# ============================================================================
# Template: Cancel Pending Order
# ============================================================================


def _build_cancel_pending_task(
    user: User,
    order: Order,
    order_id: str,
    use_email: bool,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where user wants to cancel a pending order."""
    reason = "no longer needed" if task_idx % 2 == 0 else "ordered by mistake"

    actions = []
    idx = 0

    # Authentication
    if use_email:
        actions.append(
            Action(
                action_id=_action_id("cancel_ord", idx),
                name="find_user_id_by_email",
                arguments={"email": user.email},
                compare_args=["email"],
            )
        )
    else:
        actions.append(
            Action(
                action_id=_action_id("cancel_ord", idx),
                name="find_user_id_by_name_zip",
                arguments={
                    "first_name": user.name.first_name,
                    "last_name": user.name.last_name,
                    "zip": user.address.zip,
                },
                compare_args=["first_name", "last_name", "zip"],
            )
        )
    idx += 1

    actions.append(
        Action(
            action_id=_action_id("cancel_ord", idx),
            name="get_user_details",
            arguments={"user_id": user.user_id},
            compare_args=["user_id"],
        )
    )
    idx += 1

    actions.append(
        Action(
            action_id=_action_id("cancel_ord", idx),
            name="get_order_details",
            arguments={"order_id": order_id},
            compare_args=["order_id"],
        )
    )
    idx += 1

    actions.append(
        Action(
            action_id=_action_id("cancel_ord", idx),
            name="cancel_pending_order",
            arguments={"order_id": order_id, "reason": reason},
            compare_args=["order_id", "reason"],
        )
    )

    known_info = (
        _make_user_known_info_email(user) if use_email
        else _make_user_known_info_name_zip(user)
    )

    return Task(
        id=f"synth_cancel_order_{task_idx}",
        description=Description(
            purpose="Test that agent can cancel a pending order.",
            relevant_policies="Pending orders can be cancelled with valid reason.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="retail",
                reason_for_call=f"I want to cancel my order {order_id}.",
                known_info=known_info,
                unknown_info=None,
                task_instructions=(
                    f"You want to cancel order {order_id}. "
                    f"The reason is '{reason}'. "
                    f"Confirm the cancellation when the agent explains the details."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            nl_assertions=[
                "The agent should successfully cancel the pending order.",
                "The agent should inform the user about the refund process.",
            ],
            reward_basis=[RewardType.DB, RewardType.ACTION],
        ),
    )


# ============================================================================
# Template: Exchange Delivered Items
# ============================================================================


def _build_exchange_task(
    user: User,
    order: Order,
    order_id: str,
    db: RetailDB,
    use_email: bool,
    persona: Optional[str],
    task_idx: int,
) -> Optional[Task]:
    """Build a task where user wants to exchange delivered items."""
    # Find an item in the order that has an alternative variant
    exchangeable = []
    for item in order.items:
        alt = _find_alternative_variant(db, item.product_id, item.item_id)
        if alt:
            exchangeable.append((item, alt))

    if not exchangeable:
        return None

    # Pick one item to exchange
    item_to_exchange, new_variant = exchangeable[task_idx % len(exchangeable)]
    payment_id = _get_any_payment_id(user)
    if payment_id is None:
        return None

    actions = []
    idx = 0

    # Authentication
    if use_email:
        actions.append(
            Action(
                action_id=_action_id("exchange", idx),
                name="find_user_id_by_email",
                arguments={"email": user.email},
                compare_args=["email"],
            )
        )
    else:
        actions.append(
            Action(
                action_id=_action_id("exchange", idx),
                name="find_user_id_by_name_zip",
                arguments={
                    "first_name": user.name.first_name,
                    "last_name": user.name.last_name,
                    "zip": user.address.zip,
                },
                compare_args=["first_name", "last_name", "zip"],
            )
        )
    idx += 1

    actions.extend([
        Action(
            action_id=_action_id("exchange", idx),
            name="get_user_details",
            arguments={"user_id": user.user_id},
            compare_args=["user_id"],
        ),
        Action(
            action_id=_action_id("exchange", idx + 1),
            name="get_order_details",
            arguments={"order_id": order_id},
            compare_args=["order_id"],
        ),
        Action(
            action_id=_action_id("exchange", idx + 2),
            name="get_product_details",
            arguments={"product_id": item_to_exchange.product_id},
            compare_args=["product_id"],
        ),
        Action(
            action_id=_action_id("exchange", idx + 3),
            name="exchange_delivered_order_items",
            arguments={
                "order_id": order_id,
                "item_ids": [item_to_exchange.item_id],
                "new_item_ids": [new_variant.item_id],
                "payment_method_id": payment_id,
            },
            compare_args=["order_id", "item_ids", "new_item_ids"],
        ),
    ])

    known_info = (
        _make_user_known_info_email(user) if use_email
        else _make_user_known_info_name_zip(user)
    )

    new_options_desc = _describe_item_options(new_variant.options)

    return Task(
        id=f"synth_exchange_{task_idx}",
        description=Description(
            purpose="Test that agent can exchange items in a delivered order.",
            relevant_policies="Exchange requires same product type, items must be available.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="retail",
                reason_for_call=(
                    f"I want to exchange an item in my order {order_id}."
                ),
                known_info=known_info,
                unknown_info=None,
                task_instructions=(
                    f"You want to exchange the {item_to_exchange.name} "
                    f"(currently {_describe_item_options(item_to_exchange.options)}) "
                    f"for one with options: {new_options_desc}. "
                    f"Confirm the exchange when asked."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            nl_assertions=[
                "The agent should successfully process the exchange.",
            ],
            reward_basis=[RewardType.DB, RewardType.ACTION],
        ),
    )


# ============================================================================
# Template: Return Delivered Items
# ============================================================================


def _build_return_task(
    user: User,
    order: Order,
    order_id: str,
    use_email: bool,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where user wants to return items from a delivered order."""
    # Pick item(s) to return
    num_items_to_return = min(1 + (task_idx % 2), len(order.items))
    items_to_return = order.items[:num_items_to_return]
    item_ids = [item.item_id for item in items_to_return]

    # For return, use original payment method or a gift card
    original_pm = order.payment_history[0].payment_method_id if order.payment_history else None
    payment_id = original_pm
    if payment_id is None:
        payment_id = _get_any_payment_id(user)

    if payment_id is None:
        return None

    actions = []
    idx = 0

    if use_email:
        actions.append(
            Action(
                action_id=_action_id("return", idx),
                name="find_user_id_by_email",
                arguments={"email": user.email},
                compare_args=["email"],
            )
        )
    else:
        actions.append(
            Action(
                action_id=_action_id("return", idx),
                name="find_user_id_by_name_zip",
                arguments={
                    "first_name": user.name.first_name,
                    "last_name": user.name.last_name,
                    "zip": user.address.zip,
                },
                compare_args=["first_name", "last_name", "zip"],
            )
        )
    idx += 1

    actions.extend([
        Action(
            action_id=_action_id("return", idx),
            name="get_user_details",
            arguments={"user_id": user.user_id},
            compare_args=["user_id"],
        ),
        Action(
            action_id=_action_id("return", idx + 1),
            name="get_order_details",
            arguments={"order_id": order_id},
            compare_args=["order_id"],
        ),
        Action(
            action_id=_action_id("return", idx + 2),
            name="return_delivered_order_items",
            arguments={
                "order_id": order_id,
                "item_ids": item_ids,
                "payment_method_id": payment_id,
            },
            compare_args=["order_id", "item_ids"],
        ),
    ])

    known_info = (
        _make_user_known_info_email(user) if use_email
        else _make_user_known_info_name_zip(user)
    )

    items_desc = ", ".join([f"{i.name}" for i in items_to_return])

    return Task(
        id=f"synth_return_{task_idx}",
        description=Description(
            purpose="Test that agent can process returns for delivered orders.",
            relevant_policies="Returns use original payment or gift card.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="retail",
                reason_for_call=f"I want to return some items from order {order_id}.",
                known_info=known_info,
                unknown_info=None,
                task_instructions=(
                    f"You want to return the following items: {items_desc}. "
                    f"Confirm the return when asked."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            nl_assertions=[
                "The agent should process the return successfully.",
                "The agent should inform the user about the refund timeline.",
            ],
            reward_basis=[RewardType.DB, RewardType.ACTION],
        ),
    )


# ============================================================================
# Template: Modify Pending Order Items
# ============================================================================


def _build_modify_items_task(
    user: User,
    order: Order,
    order_id: str,
    db: RetailDB,
    use_email: bool,
    persona: Optional[str],
    task_idx: int,
) -> Optional[Task]:
    """Build a task where user wants to modify items in a pending order."""
    # Find an item that has an alternative variant
    modifiable = []
    for item in order.items:
        alt = _find_alternative_variant(db, item.product_id, item.item_id)
        if alt:
            modifiable.append((item, alt))

    if not modifiable:
        return None

    item_to_modify, new_variant = modifiable[task_idx % len(modifiable)]
    payment_id = _get_any_payment_id(user)
    if payment_id is None:
        return None

    actions = []
    idx = 0

    if use_email:
        actions.append(
            Action(
                action_id=_action_id("mod_items", idx),
                name="find_user_id_by_email",
                arguments={"email": user.email},
                compare_args=["email"],
            )
        )
    else:
        actions.append(
            Action(
                action_id=_action_id("mod_items", idx),
                name="find_user_id_by_name_zip",
                arguments={
                    "first_name": user.name.first_name,
                    "last_name": user.name.last_name,
                    "zip": user.address.zip,
                },
                compare_args=["first_name", "last_name", "zip"],
            )
        )
    idx += 1

    actions.extend([
        Action(
            action_id=_action_id("mod_items", idx),
            name="get_user_details",
            arguments={"user_id": user.user_id},
            compare_args=["user_id"],
        ),
        Action(
            action_id=_action_id("mod_items", idx + 1),
            name="get_order_details",
            arguments={"order_id": order_id},
            compare_args=["order_id"],
        ),
        Action(
            action_id=_action_id("mod_items", idx + 2),
            name="get_product_details",
            arguments={"product_id": item_to_modify.product_id},
            compare_args=["product_id"],
        ),
        Action(
            action_id=_action_id("mod_items", idx + 3),
            name="modify_pending_order_items",
            arguments={
                "order_id": order_id,
                "item_ids": [item_to_modify.item_id],
                "new_item_ids": [new_variant.item_id],
                "payment_method_id": payment_id,
            },
            compare_args=["order_id", "item_ids", "new_item_ids"],
        ),
    ])

    known_info = (
        _make_user_known_info_email(user) if use_email
        else _make_user_known_info_name_zip(user)
    )

    new_options_desc = _describe_item_options(new_variant.options)

    return Task(
        id=f"synth_modify_items_{task_idx}",
        description=Description(
            purpose="Test that agent can modify items in a pending order.",
            relevant_policies="Pending orders can be modified once. Must collect all changes first.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="retail",
                reason_for_call=f"I want to change an item in my pending order {order_id}.",
                known_info=known_info,
                unknown_info=None,
                task_instructions=(
                    f"You want to change the {item_to_modify.name} "
                    f"(currently {_describe_item_options(item_to_modify.options)}) "
                    f"to one with options: {new_options_desc}. "
                    f"Confirm the modification when asked."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            nl_assertions=[
                "The agent should successfully modify the order items.",
            ],
            reward_basis=[RewardType.DB, RewardType.ACTION],
        ),
    )


# ============================================================================
# Template: Modify Pending Order Payment
# ============================================================================


def _build_modify_payment_task(
    user: User,
    order: Order,
    order_id: str,
    use_email: bool,
    persona: Optional[str],
    task_idx: int,
) -> Optional[Task]:
    """Build a task where user wants to change payment method on pending order."""
    if not order.payment_history:
        return None
    current_pm_id = order.payment_history[0].payment_method_id

    # Find a different payment method
    new_pm_id = None
    for pm_id, pm in user.payment_methods.items():
        if pm_id != current_pm_id:
            # If gift card, check balance
            if isinstance(pm, GiftCard):
                if pm.balance >= order.payment_history[0].amount:
                    new_pm_id = pm_id
                    break
            else:
                new_pm_id = pm_id
                break

    if new_pm_id is None:
        return None

    actions = []
    idx = 0

    if use_email:
        actions.append(
            Action(
                action_id=_action_id("mod_pay", idx),
                name="find_user_id_by_email",
                arguments={"email": user.email},
                compare_args=["email"],
            )
        )
    else:
        actions.append(
            Action(
                action_id=_action_id("mod_pay", idx),
                name="find_user_id_by_name_zip",
                arguments={
                    "first_name": user.name.first_name,
                    "last_name": user.name.last_name,
                    "zip": user.address.zip,
                },
                compare_args=["first_name", "last_name", "zip"],
            )
        )
    idx += 1

    actions.extend([
        Action(
            action_id=_action_id("mod_pay", idx),
            name="get_user_details",
            arguments={"user_id": user.user_id},
            compare_args=["user_id"],
        ),
        Action(
            action_id=_action_id("mod_pay", idx + 1),
            name="get_order_details",
            arguments={"order_id": order_id},
            compare_args=["order_id"],
        ),
        Action(
            action_id=_action_id("mod_pay", idx + 2),
            name="modify_pending_order_payment",
            arguments={
                "order_id": order_id,
                "payment_method_id": new_pm_id,
            },
            compare_args=["order_id", "payment_method_id"],
        ),
    ])

    known_info = (
        _make_user_known_info_email(user) if use_email
        else _make_user_known_info_name_zip(user)
    )

    return Task(
        id=f"synth_modify_payment_{task_idx}",
        description=Description(
            purpose="Test that agent can change payment method on a pending order.",
            relevant_policies="Payment can be changed on pending orders.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="retail",
                reason_for_call=f"I want to change the payment method on order {order_id}.",
                known_info=known_info,
                unknown_info=None,
                task_instructions=(
                    f"You want to switch the payment on order {order_id} "
                    f"to payment method {new_pm_id}. Confirm when asked."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            reward_basis=[RewardType.DB, RewardType.ACTION],
        ),
    )


# ============================================================================
# Template: Product Info Query
# ============================================================================


def _build_product_info_task(
    user: User,
    db: RetailDB,
    use_email: bool,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where user asks about product variants."""
    # Pick a random product
    product_ids = list(db.products.keys())
    product_id = product_ids[task_idx % len(product_ids)]
    product = db.products[product_id]

    available_count = sum(1 for v in product.variants.values() if v.available)

    actions = []
    idx = 0

    if use_email:
        actions.append(
            Action(
                action_id=_action_id("info", idx),
                name="find_user_id_by_email",
                arguments={"email": user.email},
                compare_args=["email"],
            )
        )
    else:
        actions.append(
            Action(
                action_id=_action_id("info", idx),
                name="find_user_id_by_name_zip",
                arguments={
                    "first_name": user.name.first_name,
                    "last_name": user.name.last_name,
                    "zip": user.address.zip,
                },
                compare_args=["first_name", "last_name", "zip"],
            )
        )
    idx += 1

    actions.extend([
        Action(
            action_id=_action_id("info", idx),
            name="get_product_details",
            arguments={"product_id": product_id},
            compare_args=["product_id"],
        ),
    ])

    known_info = (
        _make_user_known_info_email(user) if use_email
        else _make_user_known_info_name_zip(user)
    )

    return Task(
        id=f"synth_product_info_{task_idx}",
        description=Description(
            purpose="Test that agent can look up product information and communicate it.",
            relevant_policies="Agent should use get_product_details to answer product queries.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="retail",
                reason_for_call=f"I want to know how many {product.name} options are available.",
                known_info=known_info,
                unknown_info=None,
                task_instructions=(
                    f"Ask how many available options (variants) there are for {product.name}. "
                    f"The answer should be {available_count}."
                ),
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            communicate_info=[str(available_count)],
            reward_basis=[RewardType.COMMUNICATE],
        ),
    )


# ============================================================================
# Template: Transfer to Human (Impossible Request)
# ============================================================================


def _build_retail_transfer_task(
    user: User,
    order: Order,
    order_id: str,
    use_email: bool,
    persona: Optional[str],
    task_idx: int,
) -> Task:
    """Build a task where the user makes an impossible request."""
    impossible_requests = [
        (
            f"I want to undo the cancellation of order {order_id}.",
            "You want to reinstate a cancelled order. Insist this should be possible.",
            "cancelled",
        ),
        (
            f"I want to change the items in my processed order {order_id}.",
            "You want to modify items in an order that's already being processed.",
            "processed",
        ),
        (
            f"I want to return and exchange items from order {order_id} at the same time.",
            "You want to return some items AND exchange others from the same delivered order. "
            "This should require two separate operations but only one is allowed.",
            "delivered",
        ),
    ]

    req_idx = task_idx % len(impossible_requests)
    reason, instructions, required_status = impossible_requests[req_idx]

    actions = []
    idx = 0

    if use_email:
        actions.append(
            Action(
                action_id=_action_id("transfer", idx),
                name="find_user_id_by_email",
                arguments={"email": user.email},
                compare_args=["email"],
            )
        )
    else:
        actions.append(
            Action(
                action_id=_action_id("transfer", idx),
                name="find_user_id_by_name_zip",
                arguments={
                    "first_name": user.name.first_name,
                    "last_name": user.name.last_name,
                    "zip": user.address.zip,
                },
                compare_args=["first_name", "last_name", "zip"],
            )
        )
    idx += 1

    actions.append(
        Action(
            action_id=_action_id("transfer", idx),
            name="transfer_to_human_agents",
            arguments={"summary": "Cannot fulfill request."},
            compare_args=[],
        )
    )

    known_info = (
        _make_user_known_info_email(user) if use_email
        else _make_user_known_info_name_zip(user)
    )

    return Task(
        id=f"synth_retail_transfer_{task_idx}",
        description=Description(
            purpose="Test that agent transfers to human for impossible requests.",
        ),
        user_scenario=UserScenario(
            persona=persona,
            instructions=StructuredUserInstructions(
                domain="retail",
                reason_for_call=reason,
                known_info=known_info,
                unknown_info=None,
                task_instructions=(
                    f"Order ID is {order_id}. {instructions} "
                    f"If the agent says they can't help, ask to be transferred."
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


def verify_retail_task(task: Task) -> bool:
    """
    Verify a retail task by running golden actions on a fresh environment.
    """
    env = get_environment()
    db_before_hash = env.get_db_hash()
    actions = task.evaluation_criteria.actions or []

    write_actions = [a for a in actions if a.name not in (
        "find_user_id_by_email", "find_user_id_by_name_zip",
        "get_user_details", "get_order_details", "get_product_details",
        "list_all_product_types", "calculate", "transfer_to_human_agents",
    )]

    try:
        for action in actions:
            if action.name == "transfer_to_human_agents":
                continue
            env.make_tool_call(
                tool_name=action.name,
                requestor=action.requestor,
                **action.arguments,
            )
    except Exception as e:
        print(f"  Verification error: {e}")
        return False

    reward_basis = task.evaluation_criteria.reward_basis
    if RewardType.DB in reward_basis:
        db_after_hash = env.get_db_hash()
        if write_actions:
            if db_before_hash == db_after_hash:
                print("  DB should have changed but didn't")
                return False
        else:
            if db_before_hash != db_after_hash:
                print("  DB changed but shouldn't have")
                return False

    return True


# ============================================================================
# Main synthesis function
# ============================================================================


def create_retail_tasks(
    num_per_template: int = 5,
    seed: int = 42,
    verify: bool = True,
) -> list[Task]:
    """
    Generate synthesized retail tasks across all scenario templates.
    """
    rng = random.Random(seed)
    env = get_environment()
    db = env.tools.db
    all_tasks = []
    persona_keys = list(RETAIL_PERSONAS.keys())

    # ==== Template 1: Cancel Pending Order ====
    print("=== Generating Cancel Pending Order tasks ===")
    pending = _get_pending_orders(db)
    rng.shuffle(pending)
    for i in range(min(num_per_template, len(pending))):
        order_id, user, order = pending[i]
        use_email = (i % 2 == 0)
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_cancel_pending_task(
            user, order, order_id, use_email, RETAIL_PERSONAS[persona_key], i
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        if verify and not verify_retail_task(task):
            print(f"  ✗ {task.id} failed verification")
            continue
        all_tasks.append(task)
        print(f"  ✓ {task.id}")

    # ==== Template 2: Exchange Delivered Items ====
    print("=== Generating Exchange tasks ===")
    delivered = _get_delivered_orders(db)
    rng.shuffle(delivered)
    count = 0
    for order_id, user, order in delivered:
        if count >= num_per_template:
            break
        use_email = (count % 2 == 0)
        persona_key = persona_keys[count % len(persona_keys)]
        task = _build_exchange_task(
            user, order, order_id, db, use_email,
            RETAIL_PERSONAS[persona_key], count
        )
        if task is None:
            continue
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        if verify and not verify_retail_task(task):
            print(f"  ✗ {task.id} failed verification")
            continue
        all_tasks.append(task)
        print(f"  ✓ {task.id}")
        count += 1

    # ==== Template 3: Return Delivered Items ====
    print("=== Generating Return tasks ===")
    rng.shuffle(delivered)
    for i in range(min(num_per_template, len(delivered))):
        order_id, user, order = delivered[i]
        use_email = (i % 2 == 0)
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_return_task(
            user, order, order_id, use_email, RETAIL_PERSONAS[persona_key], i
        )
        if task is None:
            continue
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        if verify and not verify_retail_task(task):
            print(f"  ✗ {task.id} failed verification")
            continue
        all_tasks.append(task)
        print(f"  ✓ {task.id}")

    # ==== Template 4: Modify Pending Order Items ====
    print("=== Generating Modify Items tasks ===")
    rng.shuffle(pending)
    count = 0
    for order_id, user, order in pending:
        if count >= num_per_template:
            break
        use_email = (count % 2 == 0)
        persona_key = persona_keys[count % len(persona_keys)]
        task = _build_modify_items_task(
            user, order, order_id, db, use_email,
            RETAIL_PERSONAS[persona_key], count
        )
        if task is None:
            continue
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        if verify and not verify_retail_task(task):
            print(f"  ✗ {task.id} failed verification")
            continue
        all_tasks.append(task)
        print(f"  ✓ {task.id}")
        count += 1

    # ==== Template 5: Modify Pending Order Payment ====
    print("=== Generating Modify Payment tasks ===")
    rng.shuffle(pending)
    count = 0
    for order_id, user, order in pending:
        if count >= num_per_template:
            break
        use_email = (count % 2 == 0)
        persona_key = persona_keys[count % len(persona_keys)]
        task = _build_modify_payment_task(
            user, order, order_id, use_email, RETAIL_PERSONAS[persona_key], count
        )
        if task is None:
            continue
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        if verify and not verify_retail_task(task):
            print(f"  ✗ {task.id} failed verification")
            continue
        all_tasks.append(task)
        print(f"  ✓ {task.id}")
        count += 1

    # ==== Template 6: Product Info Query ====
    print("=== Generating Product Info tasks ===")
    all_users = list(db.users.values())
    rng.shuffle(all_users)
    for i in range(min(num_per_template, len(all_users))):
        user = all_users[i]
        use_email = (i % 2 == 0)
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_product_info_task(
            user, db, use_email, RETAIL_PERSONAS[persona_key], i
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        all_tasks.append(task)
        print(f"  ✓ {task.id}")

    # ==== Template 7: Transfer to Human ====
    print("=== Generating Transfer to Human tasks ===")
    # Find orders with various statuses for impossible requests
    cancelled_orders = [
        (oid, db.users[o.user_id], o) for oid, o in db.orders.items()
        if o.status == "cancelled" and o.user_id in db.users
    ]
    processed_orders = [
        (oid, db.users[o.user_id], o) for oid, o in db.orders.items()
        if o.status == "processed" and o.user_id in db.users
    ]

    transfer_sources = cancelled_orders[:2] + processed_orders[:2] + delivered[:1]
    rng.shuffle(transfer_sources)
    for i in range(min(num_per_template, len(transfer_sources))):
        order_id, user, order = transfer_sources[i]
        use_email = (i % 2 == 0)
        persona_key = persona_keys[i % len(persona_keys)]
        task = _build_retail_transfer_task(
            user, order, order_id, use_email, RETAIL_PERSONAS[persona_key], i
        )
        task.id = f"{task.id}[PERSONA:{persona_key}]"
        all_tasks.append(task)
        print(f"  ✓ {task.id}")

    print(f"\n=== Total retail tasks generated: {len(all_tasks)} ===")
    return all_tasks
