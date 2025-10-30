# Data source + intended feature contract (comments-only for now)
# RAW_SOURCE_URL: https://api.spacexdata.com/v4/launches
# RAW_EXPECTED_FIELDS:
#   ['date_utc','rocket','success','payloads','cores','flight_number','launchpad']
#
# Final FEATURE_COLUMNS (in order) — our canonical feature table:
#   - flight_number (int)
#   - year (int)
#   - payload_mass_kg (float)
#   - orbit (str: {LEO,MEO,GEO,GTO,SSO,HEO,Unknown})
#   - launch_site (str)
#   - booster_version (str)
#   - reuse_count (int)
#   - is_weekend (int: 0/1)
#   - success (int: 0/1)
#
# These columns + their order become the feature manifest for training/serving.
