from django.apps import AppConfig
import dramatiq
from dramatiq_abort import Abortable, backends
from django.conf import settings
# from dramatiq.brokers.redis import RedisBroker


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"


# redis_broker = RedisBroker(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
# dramatiq.set_broker(redis_broker)

event_backend = backends.RedisBackend.from_url(settings.REDIS_URL)
abortable = Abortable(backend=event_backend)
dramatiq.get_broker().add_middleware(abortable)
