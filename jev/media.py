"""Bounded inline image transport; no remote fetching on the API server."""
import base64
import binascii

# Initial transport budget: one 4 MiB raw image, inside an 8 MiB HTTP body.
# Keeps base64/request copies bounded in the existing 1 GiB, two-request adapter.
MAX_IMAGE_BYTES = 4 * 1024 * 1024
BODY_LIMIT = 8 * 1024 * 1024
SIGNATURES = {'image/png': b'\x89PNG\r\n\x1a\n', 'image/jpeg': b'\xff\xd8\xff'}


def encode_image(data):
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError('image must be 1..4 MiB')
    mime = next((name for name, signature in SIGNATURES.items() if data.startswith(signature)), None)
    if mime is None:
        raise ValueError('image must be PNG or JPEG')
    return 'data:' + mime + ';base64,' + base64.b64encode(data).decode('ascii')


def validate_image(value):
    if not isinstance(value, str):
        raise ValueError('image must be an inline PNG or JPEG data URL')
    prefix, separator, encoded = value.partition(',')
    mime = prefix.removeprefix('data:').removesuffix(';base64')
    if not separator or prefix != 'data:' + mime + ';base64' or mime not in SIGNATURES:
        raise ValueError('image must be an inline PNG or JPEG data URL')
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ValueError('image exceeds 4 MiB')
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError('invalid image base64') from error
    if not data or len(data) > MAX_IMAGE_BYTES or not data.startswith(SIGNATURES[mime]):
        raise ValueError('invalid image format or size')
    return value
