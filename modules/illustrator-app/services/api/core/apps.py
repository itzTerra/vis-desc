from django.apps import AppConfig
import dramatiq
from dramatiq_abort import Abortable, backends
from django.conf import settings


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"


event_backend = backends.RedisBackend.from_url(settings.REDIS_URL)
abortable = Abortable(backend=event_backend)
dramatiq.get_broker().add_middleware(abortable)
