from email.parser import Parser

def get_payload(message):
    """Retrieves payload from email as text; recurses on multi-part messages"""
    payload = message.get_payload()
    if isinstance(payload, list):
        return '\n'.join(get_payload(submessage) for submessage in payload)
    else:
        return payload

def read(fp):
    """Reads message from fp; returns message body as string"""
    message = Parser().parse(fp)
    return get_payload(message)
