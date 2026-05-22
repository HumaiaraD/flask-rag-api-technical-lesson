"""Facility operations documents for the Flask RAG API lesson."""

from typing import Dict, List


FACILITY_DOCUMENTS: List[Dict[str, str]] = [
    {
        "id": "FAC-101",
        "title": "Replacing a Lost or Damaged Employee Badge",
        "category": "access",
        "text": (
            "Employees who lose or damage a badge should submit a badge replacement "
            "request through the facilities portal. Temporary badges are available at "
            "the front desk after identity verification."
        ),
    },
    {
        "id": "FAC-102",
        "title": "After-Hours Building Access",
        "category": "access",
        "text": (
            "Employees may enter the building after 7 PM only when after-hours access "
            "has been enabled on their badge. Requests should be submitted before the "
            "needed date and approved by the employee's manager."
        ),
    },
    {
        "id": "FAC-103",
        "title": "Conference Room Setup Requests",
        "category": "events",
        "text": (
            "Room setup requests for meetings, trainings, or client visits should be "
            "submitted at least two business days in advance. Include room name, head "
            "count, seating layout, audio needs, and any equipment requirements."
        ),
    },
    {
        "id": "FAC-104",
        "title": "Temperature and Maintenance Requests",
        "category": "maintenance",
        "text": (
            "Heating, cooling, lighting, plumbing, or furniture issues should be "
            "reported through a facilities maintenance ticket. Include the floor, room "
            "number, issue description, urgency, and a photo when helpful."
        ),
    },
    {
        "id": "FAC-105",
        "title": "Visitor Registration and Lobby Check-In",
        "category": "visitors",
        "text": (
            "Visitors must be registered before arrival. The host should add the guest "
            "name, company, visit date, and host contact information. Guests receive a "
            "temporary visitor badge at lobby check-in."
        ),
    },
    {
        "id": "FAC-106",
        "title": "Office Equipment Repair Requests",
        "category": "maintenance",
        "text": (
            "Broken shared equipment, including printers, monitors, projectors, and "
            "badge readers, should be reported with the asset name, location, error "
            "message, and a description of the problem."
        ),
    },
]
