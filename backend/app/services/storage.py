import os
import base64
import hashlib
import asyncio
from typing import Dict, Optional
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.case import CaseEncryptionKey

# In-memory session cache for decrypted DEKs to avoid unneeded cryptographic unwrap overhead
_DEK_CACHE: Dict[str, bytes] = {}
_DEK_LOCK = asyncio.Lock()


def get_master_kek() -> bytes:
    """
    Derives a 256-bit (32-byte) Master Key Encryption Key (KEK) from configured settings.
    """
    raw_key = settings.STORAGE_MASTER_KEY.encode("utf-8")
    return hashlib.sha256(raw_key).digest()


async def get_or_create_case_dek(case_id: str, db: Optional[AsyncSession] = None) -> bytes:
    """
    Retrieves or generates a per-case Data Encryption Key (DEK).
    Envelope Encryption:
    - Master KEK (32 bytes) wraps the per-case DEK using AES-256-GCM.
    - Wrapped DEK is persisted in the database table 'case_encryption_keys'.
    - Unwrapped DEK is cached in memory per case.
    """
    if case_id in _DEK_CACHE:
        return _DEK_CACHE[case_id]

    async with _DEK_LOCK:
        if case_id in _DEK_CACHE:
            return _DEK_CACHE[case_id]

        if db is None:
            raise ValueError(f"DEK for case {case_id} not cached and db session not provided.")

        master_kek = get_master_kek()

        stmt = select(CaseEncryptionKey).where(CaseEncryptionKey.case_id == case_id)
        result = await db.execute(stmt)
        key_record = result.scalar_one_or_none()

        if key_record:
            # Unwrap existing DEK
            raw_wrapped = base64.b64decode(key_record.encrypted_dek)
            nonce = raw_wrapped[:12]
            ciphertext = raw_wrapped[12:]
            aesgcm = AESGCM(master_kek)
            dek = aesgcm.decrypt(nonce, ciphertext, associated_data=case_id.encode("utf-8"))
        else:
            # Generate new 256-bit DEK
            dek = os.urandom(32)
            nonce = os.urandom(12)
            aesgcm = AESGCM(master_kek)
            ciphertext = aesgcm.encrypt(nonce, dek, associated_data=case_id.encode("utf-8"))
            wrapped_str = base64.b64encode(nonce + ciphertext).decode("utf-8")

            new_key = CaseEncryptionKey(
                case_id=case_id,
                encrypted_dek=wrapped_str
            )
            db.add(new_key)
            await db.flush()

        _DEK_CACHE[case_id] = dek
        return dek


def encrypt_payload(dek: bytes, case_id: str, data: bytes) -> bytes:
    """
    Encrypts arbitrary data using AES-256-GCM with the per-case DEK.
    Output: 12-byte Nonce + Ciphertext (includes 16-byte authentication tag).
    """
    nonce = os.urandom(12)
    aesgcm = AESGCM(dek)
    ciphertext = aesgcm.encrypt(nonce, data, associated_data=case_id.encode("utf-8"))
    return nonce + ciphertext


def decrypt_payload(dek: bytes, case_id: str, encrypted_blob: bytes) -> bytes:
    """
    Decrypts an AES-256-GCM payload using the per-case DEK and verifies authentication tag.
    """
    if len(encrypted_blob) < 28:
        raise ValueError("Encrypted payload too short to contain valid nonce and GCM auth tag.")
    nonce = encrypted_blob[:12]
    ciphertext = encrypted_blob[12:]
    aesgcm = AESGCM(dek)
    return aesgcm.decrypt(nonce, ciphertext, associated_data=case_id.encode("utf-8"))


class EncryptedStorageService:
    """
    Encrypted Object Storage service for forensic image artifacts.
    Supports local encrypted directory (dev) and MinIO/S3 compatible storage with AES-256-GCM envelope encryption.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or settings.STORAGE_LOCAL_DIR)

    def _get_absolute_path(self, relative_storage_path: str) -> Path:
        return self.base_dir / relative_storage_path

    async def store_image(
        self,
        case_id: str,
        image_id: str,
        image_bytes: bytes,
        db: Optional[AsyncSession] = None
    ) -> str:
        """
        Envelope encrypts and persists an image to storage.
        Returns the relative storage path.
        """
        dek = await get_or_create_case_dek(case_id, db)
        encrypted_blob = encrypt_payload(dek, case_id, image_bytes)

        relative_path = f"{case_id}/{image_id}.enc"
        full_path = self._get_absolute_path(relative_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, "wb") as f:
            f.write(encrypted_blob)

        return relative_path

    async def retrieve_image(
        self,
        case_id: str,
        relative_storage_path: str,
        db: Optional[AsyncSession] = None
    ) -> bytes:
        """
        Retrieves and decrypts an image using the case DEK.
        """
        full_path = self._get_absolute_path(relative_storage_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Encrypted image artifact not found: {relative_storage_path}")

        with open(full_path, "rb") as f:
            encrypted_blob = f.read()

        dek = await get_or_create_case_dek(case_id, db)
        return decrypt_payload(dek, case_id, encrypted_blob)

    async def save_image(
        self,
        case_id: str,
        image_bytes: bytes,
        version: int,
        is_candidate: bool = False,
        image_id: Optional[str] = None,
        db: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """Convenience helper to store an image and return metadata dict."""
        import uuid
        img_id = image_id or str(uuid.uuid4())
        relative_path = await self.store_image(case_id, img_id, image_bytes, db=db)
        return {
            "image_id": img_id,
            "storage_path": relative_path,
            "url": f"/api/v1/cases/{case_id}/images/{img_id}"
        }

    async def load_image(
        self,
        relative_storage_path: str,
        case_id: Optional[str] = None,
        db: Optional[AsyncSession] = None
    ) -> bytes:
        """Convenience helper to retrieve an image by relative storage path."""
        if not case_id and "/" in relative_storage_path:
            case_id = relative_storage_path.split("/")[0]
        if not case_id:
            raise ValueError("case_id must be provided or inferrable from relative_storage_path.")
        return await self.retrieve_image(case_id, relative_storage_path, db=db)


# Global singleton instance
storage_service = EncryptedStorageService()
