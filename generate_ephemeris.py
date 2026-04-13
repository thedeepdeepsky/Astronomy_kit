import json
from datetime import datetime
from SkyAstrokit import ephemeris

# Generate ephemeris data
data = ephemeris.get_sunrise_sunset()

# Save to file
with open('ephemeris_data.json', 'w') as f:
    json.dump(data, f, indent=2, default=str)

print("Ephemeris data generated successfully!")
