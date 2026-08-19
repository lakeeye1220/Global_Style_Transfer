
import os
import huggingface_hub as _hfh
import huggingface_hub.constants as _hfh_constants 

# 1. hf_cache_home
if not hasattr(_hfh_constants, "hf_cache_home"):
    _hfh_constants.hf_cache_home = os.path.dirname(_hfh_constants.HUGGINGFACE_HUB_CACHE)

# 2. HfFolder 
if not hasattr(_hfh, "HfFolder"):
    class HfFolder:
        @staticmethod
        def get_token():
            return _hfh.get_token()

        @staticmethod
        def save_token(token):
            return _hfh.login(token=token, add_to_git_credential=False)

        @staticmethod
        def delete_token():
            return _hfh.logout()

    _hfh.HfFolder = HfFolder

# 3. cached_download 
if not hasattr(_hfh, "cached_download"):
    def cached_download(url_or_filename=None, cache_dir=None, force_filename=None,
                         proxies=None, resume_download=False, force_download=False,
                         user_agent=None, use_auth_token=None, **kwargs):
        return _hfh.hf_hub_download(
            repo_id=kwargs.get("repo_id", url_or_filename),
            filename=kwargs.get("filename"),
            cache_dir=cache_dir,
            force_filename=force_filename,
            proxies=proxies,
            resume_download=resume_download,
            force_download=force_download,
            token=use_auth_token,
        )

    _hfh.cached_download = cached_download

print("[hf_compat] huggingface_hub")