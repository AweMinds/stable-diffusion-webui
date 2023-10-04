# represent a user
# each user has a standalone work dir
import re


class User:
    def __init__(self, uid, gid, tire):
        self._uid = uid
        self._gid = gid
        self._tire = tire

    @property
    def uid(self):
        return self._uid

    @property
    def tire(self):
        return self._tire

    @classmethod
    def current_user(cls, request):
        uid = ''
        tire = ''
        if request:
            if 'cookie' in request.headers:
                cookie_ = request.headers['cookie']
                user_id_pattern = r"user-id=([\w|-]+)"
                user_tire_pattern = r"user-tire=([\w|-]+)"
                user_id_match = re.search(user_id_pattern, cookie_)
                user_tire_match = re.search(user_tire_pattern, cookie_)

                if user_id_match:
                    uid = user_id_match.group(1)
                if user_tire_match:
                    tire = user_tire_match.group(1)
        if not uid:
            # consider user as anonymous if User-Id is not present in request headers
            uid = 'anonymous'
        return User(uid, '', tire)
