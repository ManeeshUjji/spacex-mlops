from typing import Dict

def make_explanation(x: Dict) -> str:
    bits = []
    try:
        orb = str(x.get("orbit", "")).upper()
        if orb in {"GTO", "GEO", "SSO"}:
            bits.append("High-energy orbit can reduce landing odds")
        elif orb in {"LEO", "ISS"}:
            bits.append("Lower-energy orbit is favorable")

        pm = float(x.get("payload_mass_kg", 0) or 0)
        if pm > 10000:
            bits.append("Heavy payload increases difficulty")
        elif pm < 3000:
            bits.append("Light payload improves odds")

        bv = str(x.get("booster_version", "")).lower()
        if "block 5" in bv or "block5" in bv:
            bits.append("Block 5 booster improves reliability")
        elif "block 3" in bv or "block 4" in bv:
            bits.append("Older booster may reduce odds")

        rc = int(x.get("reuse_count", 0) or 0)
        if rc >= 3:
            bits.append("Higher reuse count suggests proven hardware")
        elif rc == 0:
            bits.append("New core without flight history")

        return "; ".join(bits) if bits else "Standard conditions; outcome driven by combined factors"
    except Exception:
        return "Standard conditions; outcome driven by combined factors"
