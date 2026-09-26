from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid
try:
    from backend.database import db_get_user, db_create_user
except ImportError:
    from database import db_get_user, db_create_user

router = APIRouter(prefix="/api/auth", tags=["auth"])

class LoginReq(BaseModel):
    email: str
    password: str

@router.post("/login")
def login(req: LoginReq):
    user = db_get_user(req.email, req.password)
    if user:
        return {
            "success": True, 
            "token": user["id"], 
            "user": {
                "id": user["id"],
                "email": user["email"],
                "full_name": user["full_name"],
                "organization": user["organization"],
                "role": user.get("role", "client")
            }
        }
    raise HTTPException(status_code=401, detail="Sai email hoặc mật khẩu")

class RegisterReq(BaseModel):
    email: str
    password: str
    full_name: str
    organization: str

@router.post("/register")
def register(req: RegisterReq):
    existing = db_get_user(req.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email đã được sử dụng")
        
    user_id = f"usr_{str(uuid.uuid4())[:8]}"
    try:
        db_create_user(user_id, req.email, req.password, req.full_name, req.organization, 'client')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tạo tài khoản: {str(e)}")
        
    return {"success": True, "message": "Đăng ký thành công", "user_id": user_id}
