import redis
from django.conf import settings


def get_redis_client():
    return redis.from_url(settings.REDIS_URL)
