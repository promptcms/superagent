from langfuse.callback import CallbackHandler
from decouple import config

PUBLIC_KEY = config("LANGFUSE_PUBLIC_KEY")
SECRET_KEY = config("LANGFUSE_SECRET_KEY")

langfuse_handler = CallbackHandler(PUBLIC_KEY, SECRET_KEY)
