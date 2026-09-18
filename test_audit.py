from app.core.database import SessionLocal
from app.services.audit_service import AuditService


db = SessionLocal()

try:
    log = AuditService.log(
        db=db,
        event_type="FACE_LOGIN_SUCCESS",
        success=True,
        details="Audit logging system test",
    )

    print("Audit log created successfully")
    print("ID:", log.id)
    print("Event:", log.event_type)
    print("Success:", log.success)

finally:
    db.close()