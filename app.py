
import streamlit as st
import os
import json
import hashlib
import re
import phonenumbers
import time
import threading
import numpy as np

from PIL import Image, ImageDraw
from io import BytesIO
from pathlib import Path

from ai_recognition import load_ai_model, compare_images, extract_features

# Live camera support is optional at import time. Install the dependency
# from requirements.txt to enable the Live Auto Scan mode.
try:
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    WEBRTC_AVAILABLE = True
    WEBRTC_IMPORT_ERROR = ""
except ImportError as import_error:
    av = None
    webrtc_streamer = None
    WebRtcMode = None
    WEBRTC_AVAILABLE = False
    WEBRTC_IMPORT_ERROR = str(import_error)


# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Cow Nose Identification System",
    page_icon="🐄",
    layout="wide"
)

DATA_FILE = "cows.json"
IMAGE_FOLDER = "cow_images"

os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Conservative prototype settings. These must be calibrated with a labelled
# validation set of genuine matches and non-matching cows before real use.
AI_THRESHOLD = 0.75
AI_MIN_COW_MARGIN = 0.06


# =====================================================
# AI MODEL
# =====================================================

@st.cache_resource
def initialize_ai():
    load_ai_model()
    return True


try:
    initialize_ai()
    AI_READY = True
except Exception as error:
    AI_READY = False
    AI_ERROR = str(error)


# =====================================================
# DATA FUNCTIONS
# =====================================================

def load_cows():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    return []


def save_cows(cows):
    temp_file = DATA_FILE + ".tmp"

    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(cows, file, indent=4, ensure_ascii=False)

    os.replace(temp_file, DATA_FILE)


def get_image_hash(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()


def safe_filename(value):
    value = re.sub(r"[^a-zA-Z0-9_-]", "_", value.strip())
    return value[:60] or "cow"


def get_cow_images(cow):
    """
    Supports both new multi-image records
    and old single-image records.
    """
    images = cow.get("images", [])

    if isinstance(images, str):
        images = [images]

    old_image = cow.get("image")

    if old_image and old_image not in images:
        images.append(old_image)

    return [
        path for path in images
        if path and os.path.isfile(path)
    ]


def save_uploaded_images(cow_id, uploaded_images):
    saved_paths = []
    safe_id = safe_filename(cow_id)

    try:
        for index, uploaded in enumerate(uploaded_images, start=1):
            image = Image.open(uploaded).convert("RGB")

            filename = f"{safe_id}_{index}.jpg"
            path = os.path.join(IMAGE_FOLDER, filename)

            image.save(path, format="JPEG", quality=95)
            saved_paths.append(path)

        return saved_paths

    except Exception:
        for path in saved_paths:
            if os.path.exists(path):
                os.remove(path)
        raise


# =====================================================
# COUNTRY AND PHONE VALIDATION
# =====================================================

COUNTRIES = [
    ("India", "IN", "+91"),
    ("United States", "US", "+1"),
    ("United Kingdom", "GB", "+44"),
    ("Canada", "CA", "+1"),
    ("Australia", "AU", "+61"),
    ("United Arab Emirates", "AE", "+971"),
    ("Nepal", "NP", "+977"),
    ("Bangladesh", "BD", "+880"),
    ("Pakistan", "PK", "+92"),
    ("Sri Lanka", "LK", "+94"),
    ("Saudi Arabia", "SA", "+966"),
    ("Singapore", "SG", "+65"),
    ("Germany", "DE", "+49"),
    ("France", "FR", "+33"),
    ("Japan", "JP", "+81"),
    ("China", "CN", "+86"),
    ("South Africa", "ZA", "+27"),
    ("New Zealand", "NZ", "+64"),
    ("Brazil", "BR", "+55"),
    ("Russia", "RU", "+7")
]


def validate_phone(number, region):
    try:
        parsed = phonenumbers.parse(number, region)

        if not phonenumbers.is_possible_number(parsed):
            return False, "Mobile number length is invalid."

        if not phonenumbers.is_valid_number(parsed):
            return False, "Please enter a valid mobile number."

        digits = str(parsed.national_number)

        if len(digits) < 10:
            return False, "Mobile number must contain at least 10 digits."

        formatted = phonenumbers.format_number(
            parsed,
            phonenumbers.PhoneNumberFormat.INTERNATIONAL
        )

        return True, formatted

    except phonenumbers.NumberParseException:
        return False, "Invalid mobile number."


# =====================================================
# AI RECOGNITION
# =====================================================

def ai_similarity(image1, image2):
    """
    Uses DINOv2 comparison function from ai_recognition.py.
    Returns a similarity score between 0 and 1.
    """
    try:
        image1 = image1.convert("RGB")
        image2 = image2.convert("RGB")

        score = compare_images(image1, image2)

        return max(0.0, min(1.0, float(score)))

    except Exception as error:
        st.warning(f"AI comparison error: {error}")
        return 0.0


def rank_cow_matches(scored_references):
    """Rank cows, not individual photos, so one unusually similar photo
    cannot by itself dominate all other registered records.

    Each item in scored_references is (reference_dict, cosine_score).
    When a cow has multiple registered images, its score blends its best
    score (70%) with its second-best score (30%).
    """
    grouped = {}

    for reference, score in scored_references:
        cow = reference["cow"]
        cow_key = (
            str(cow.get("id") or "").strip()
            or str(cow.get("name") or "").strip()
            or str(id(cow))
        )

        if cow_key not in grouped:
            grouped[cow_key] = {
                "cow": cow,
                "cow_key": cow_key,
                "scores": [],
                "best_reference_score": -1.0,
                "matched_image": reference.get("image_path"),
            }

        entry = grouped[cow_key]
        score = float(score)
        entry["scores"].append(score)

        if score > entry["best_reference_score"]:
            entry["best_reference_score"] = score
            entry["matched_image"] = reference.get("image_path")

    ranked = []
    for entry in grouped.values():
        ordered_scores = sorted(entry["scores"], reverse=True)
        if len(ordered_scores) >= 2:
            combined_score = (
                0.70 * ordered_scores[0]
                + 0.30 * ordered_scores[1]
            )
        else:
            combined_score = ordered_scores[0]

        ranked.append({
            "cow": entry["cow"],
            "cow_key": entry["cow_key"],
            "score": combined_score,
            "matched_image": entry["matched_image"],
            "reference_count": len(ordered_scores),
        })

    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def identify_cow_ai(query_image, cows):
    """Compare an image against registered cows and reject weak or ambiguous
    matches using a minimum score and a best-vs-runner-up margin.
    """
    scored_references = []
    query_image = query_image.convert("RGB")

    for cow in cows:
        for image_path in get_cow_images(cow):
            try:
                with Image.open(image_path) as registered_image:
                    score = ai_similarity(
                        query_image,
                        registered_image.convert("RGB")
                    )

                scored_references.append(({
                    "cow": cow,
                    "image_path": image_path,
                }, score))

            except Exception as error:
                print(f"Could not process {image_path}: {error}")

    ranked = rank_cow_matches(scored_references)
    if not ranked:
        return None, 0.0, None

    best = ranked[0]
    second_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
    margin = best["score"] - second_score

    if best["score"] >= AI_THRESHOLD and margin >= AI_MIN_COW_MARGIN:
        return best["cow"], best["score"], best["matched_image"]

    return None, best["score"], best["matched_image"]


def show_match_details(cow, score, matched_image):
    st.success("AI Matching Cow Found!")

    st.metric(
        "AI Similarity Score",
        f"{score * 100:.2f}%"
    )
    st.caption("Similarity score only; it is not a verified identification probability or accuracy percentage.")

    st.subheader("Registered Cow Details")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Cow ID:**", cow.get("id", "N/A"))
        st.write("**Cow Name:**", cow.get("name", "N/A"))
        st.write("**Age:**", cow.get("age", "N/A"))
        st.write("**Breed:**", cow.get("breed", "N/A"))

    with col2:
        st.write("**Owner:**", cow.get("owner_name", "N/A"))
        st.write("**Contact:**", cow.get("contact", "N/A"))
        st.write("**Country:**", cow.get("country", "N/A"))
        st.write("**Address:**", cow.get("address", "N/A"))

    if matched_image and os.path.exists(matched_image):
        st.image(
            matched_image,
            caption="Best Matching Registered Nose Image",
            width=300
        )


def run_ai_identification(uploaded_image, cows):
    if not AI_READY:
        st.error(f"AI model could not load: {AI_ERROR}")
        return

    if not cows:
        st.warning("No registered cows available.")
        return

    with st.spinner("AI is analyzing cow nose patterns..."):
        query_image = Image.open(
            BytesIO(uploaded_image.getvalue())
        ).convert("RGB")

        cow, score, matched_image = identify_cow_ai(
            query_image,
            cows
        )

    if cow:
        show_match_details(cow, score, matched_image)

    else:
        st.error("No Matching Cow Found.")

        st.write(
            f"Best similarity score: {score * 100:.2f}%"
        )

        st.caption(
            "No reliable match was accepted. The best cow must reach at least "
            f"{AI_THRESHOLD * 100:.0f}% similarity and lead the second-best "
            f"cow by at least {AI_MIN_COW_MARGIN * 100:.0f} percentage points. "
            "These are provisional prototype settings, not validated accuracy limits."
        )


# =====================================================
# LIVE CAMERA SCAN HELPERS
# =====================================================

LIVE_SCAN_INTERVAL_SECONDS = 1.5
LIVE_CONFIRMATION_FRAMES = 3
LIVE_GUIDE_MARGIN = 0.15


class LiveScanState:
    """Thread-safe state shared by the WebRTC callback and Streamlit UI."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            "status": "waiting",
            "cow": None,
            "score": 0.0,
            "matched_image": None,
            "message": "Start the camera to begin scanning.",
        }

    def update(self, status, cow=None, score=0.0, matched_image=None, message=""):
        with self._lock:
            self._data = {
                "status": status,
                "cow": cow,
                "score": float(score),
                "matched_image": matched_image,
                "message": message,
            }

    def snapshot(self):
        with self._lock:
            return dict(self._data)

    def reset(self, message="Start the camera to begin scanning."):
        self.update("waiting", message=message)


@st.cache_data(show_spinner=False)
def get_cached_image_embedding(image_path, modified_time_ns):
    """Extract and cache a registered image embedding until that file changes."""
    with Image.open(image_path) as registered_image:
        embedding = extract_features(registered_image.convert("RGB"))

    return embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)


def prepare_live_reference_embeddings(cow_records):
    """Prepare the registered nose embeddings once before live scanning."""
    references = []

    for cow in cow_records:
        for image_path in get_cow_images(cow):
            try:
                file_stat = os.stat(image_path)
                embedding = get_cached_image_embedding(
                    image_path,
                    file_stat.st_mtime_ns
                )

                references.append({
                    "embedding": embedding,
                    "cow": cow,
                    "image_path": image_path,
                })

            except Exception as error:
                print(f"Could not prepare live reference {image_path}: {error}")

    return references


def create_live_video_callback(reference_embeddings, scan_state):
    """Build an interval-based WebRTC callback with repeated-match confirmation."""
    last_scan_time = 0.0
    previous_candidate_id = None
    consecutive_matches = 0

    def video_frame_callback(frame):
        nonlocal last_scan_time, previous_candidate_id, consecutive_matches

        frame_bgr = frame.to_ndarray(format="bgr24")
        height, width = frame_bgr.shape[:2]

        # The user should position the cow nose inside this central guide.
        x0 = int(width * LIVE_GUIDE_MARGIN)
        y0 = int(height * LIVE_GUIDE_MARGIN)
        x1 = int(width * (1 - LIVE_GUIDE_MARGIN))
        y1 = int(height * (1 - LIVE_GUIDE_MARGIN))

        now = time.monotonic()
        current_status = scan_state.snapshot()["status"]
        # Once a match is accepted, keep the result visible until the user stops
        # the camera. This prevents a later noisy frame from replacing the result.
        if current_status != "matched" and now - last_scan_time >= LIVE_SCAN_INTERVAL_SECONDS:
            last_scan_time = now

            try:
                if not reference_embeddings:
                    scan_state.update(
                        "error",
                        message="No registered nose-image embeddings are available."
                    )
                else:
                    # Compare only the central guide area to reduce background influence.
                    nose_region_bgr = frame_bgr[y0:y1, x0:x1]
                    nose_region_rgb = nose_region_bgr[:, :, ::-1].copy()
                    query_image = Image.fromarray(nose_region_rgb).convert("RGB")

                    query_embedding = extract_features(query_image)
                    query_vector = (
                        query_embedding.squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                    scored_references = []
                    for reference in reference_embeddings:
                        # DINOv2 embeddings are L2-normalized, so dot product
                        # is equivalent to cosine similarity here.
                        score = float(np.dot(query_vector, reference["embedding"]))
                        score = max(-1.0, min(1.0, score))
                        scored_references.append((reference, score))

                    ranked = rank_cow_matches(scored_references)

                    if ranked:
                        best = ranked[0]
                        second_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
                        display_score = max(0.0, min(1.0, best["score"]))
                        score_margin = display_score - second_score

                        if (
                            display_score >= AI_THRESHOLD
                            and score_margin >= AI_MIN_COW_MARGIN
                        ):
                            candidate_cow = best["cow"]
                            candidate_id = best["cow_key"]

                            if candidate_id == previous_candidate_id:
                                consecutive_matches += 1
                            else:
                                previous_candidate_id = candidate_id
                                consecutive_matches = 1

                            if consecutive_matches >= LIVE_CONFIRMATION_FRAMES:
                                scan_state.update(
                                    "matched",
                                    cow=candidate_cow,
                                    score=display_score,
                                    matched_image=best["matched_image"],
                                    message=(
                                        "The same cow passed the score and "
                                        "best-vs-second-best checks in "
                                        f"{LIVE_CONFIRMATION_FRAMES} consecutive scans."
                                    ),
                                )
                            else:
                                scan_state.update(
                                    "confirming",
                                    cow=candidate_cow,
                                    score=display_score,
                                    matched_image=best["matched_image"],
                                    message=(
                                        f"Possible match: {consecutive_matches}/"
                                        f"{LIVE_CONFIRMATION_FRAMES} confirmations. "
                                        "Keep the nose steady inside the guide."
                                    ),
                                )
                        else:
                            previous_candidate_id = None
                            consecutive_matches = 0

                            if display_score >= AI_THRESHOLD and score_margin < AI_MIN_COW_MARGIN:
                                scan_state.update(
                                    "ambiguous",
                                    score=display_score,
                                    message=(
                                        "No reliable match: the top two cows are "
                                        f"too close ({score_margin * 100:.1f}% score gap). "
                                        "Adjust the nose position and scan again."
                                    ),
                                )
                            else:
                                scan_state.update(
                                    "searching",
                                    score=display_score,
                                    message=(
                                        "No reliable match yet. Best score "
                                        f"{display_score * 100:.1f}%; required "
                                        f"{AI_THRESHOLD * 100:.0f}% plus a clear "
                                        "lead over the next cow."
                                    ),
                                )
                    else:
                        previous_candidate_id = None
                        consecutive_matches = 0
                        scan_state.update(
                            "searching",
                            score=0.0,
                            message="No registered cow reference images are available.",
                        )

            except Exception as error:
                previous_candidate_id = None
                consecutive_matches = 0
                scan_state.update(
                    "error",
                    message=f"Live scan error: {error}"
                )

        # Draw a simple framing guide on the returned preview.
        preview_rgb = Image.fromarray(frame_bgr[:, :, ::-1].copy())
        draw = ImageDraw.Draw(preview_rgb)
        draw.rectangle((x0, y0, x1, y1), outline=(0, 220, 120), width=4)
        preview_bgr = np.asarray(preview_rgb)[:, :, ::-1].copy()
        return av.VideoFrame.from_ndarray(preview_bgr, format="bgr24")

    return video_frame_callback


def get_live_rtc_configuration():
    """Use public STUN by default; optional TURN credentials can be added in Streamlit Secrets."""
    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]

    try:
        turn_url = st.secrets.get("TURN_SERVER_URL", "")
        turn_username = st.secrets.get("TURN_SERVER_USERNAME", "")
        turn_credential = st.secrets.get("TURN_SERVER_CREDENTIAL", "")

        if turn_url and turn_username and turn_credential:
            ice_servers.append({
                "urls": [turn_url],
                "username": turn_username,
                "credential": turn_credential,
            })
    except Exception:
        # Secrets are optional; STUN-only mode remains available for networks
        # that do not require a TURN relay.
        pass

    return {"iceServers": ice_servers}


@st.fragment(run_every=1)
def render_live_scan_result(scan_state):
    """Refresh scan status; request a full rerun when a match stops the camera."""
    result = scan_state.snapshot()
    status = result["status"]

    if status == "matched":
        # desired_playing_state is applied by the parent app on a full rerun.
        # Turn it off as soon as the callback reports a confirmed match.
        if st.session_state.get("live_camera_enabled", False):
            st.session_state.live_camera_enabled = False
            st.rerun(scope="app")
        st.success("Cow match found. Camera is stopping automatically...")
    elif status == "confirming":
        st.info(result["message"])
        st.metric("Current Similarity Score", f"{result['score'] * 100:.2f}%")
    elif status == "searching":
        st.info(result["message"])
        st.metric("Best Similarity Score", f"{result['score'] * 100:.2f}%")
    elif status == "ambiguous":
        st.warning(result["message"])
        st.metric("Best Similarity Score", f"{result['score'] * 100:.2f}%")
    elif status == "error":
        st.error(result["message"])
    else:
        st.info(result["message"])


# =====================================================
# APP HEADER
# =====================================================

st.title("🐄 Cow Nose Identification System")

st.caption(
    "Contactless Biometric Identification and Traceability System"
)

st.divider()

if AI_READY:
    st.success("DINOv2 AI Recognition Model Loaded Successfully")
else:
    st.error(f"AI Model Error: {AI_ERROR}")

cows = load_cows()


# =====================================================
# TABS
# =====================================================

register_tab, identify_tab, camera_tab, records_tab = st.tabs(
    [
        "Register Cow",
        "Identify Cow",
        "Camera Scan",
        "Registered Records"
    ]
)


# =====================================================
# REGISTER COW
# =====================================================

with register_tab:

    st.header("Register New Cow")

    with st.form("register_cow_form"):

        col1, col2 = st.columns(2)

        with col1:

            cow_id = st.text_input("Cow ID *")

            cow_name = st.text_input("Cow Name *")

            age = st.text_input(
                "Age (e.g. 3 years) *"
            )

            breed = st.text_input("Breed *")

        with col2:

            owner_name = st.text_input("Owner Name *")

            st.markdown("Contact Number *")

            country_options = [
                f"{name} ({code})"
                for name, region, code in COUNTRIES
            ]

            selected_country = st.selectbox(
                "Select Country",
                country_options,
                index=0
            )

            country_index = country_options.index(selected_country)
            country_name, country_region, country_code = COUNTRIES[country_index]

            st.caption(f"Country calling code: {country_code}")

            phone_number = st.text_input(
                "Mobile Number *",
                placeholder="1234567890"
            )

            address = st.text_area("Address *")

        st.markdown("**Upload Cow Nose Images ***")

        cow_images = st.file_uploader(
            "Select one or more nose images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if cow_images:
            st.write(f"Selected images: {len(cow_images)}")

            preview_cols = st.columns(4)

            for index, uploaded in enumerate(cow_images):
                with preview_cols[index % 4]:
                    st.image(
                        uploaded,
                        caption=f"Image {index + 1}",
                        use_container_width=True
                    )

        submit = st.form_submit_button(
            "Register Cow",
            type="primary",
            use_container_width=True
        )

        if submit:

            required_fields = {
                "Cow ID": cow_id,
                "Cow Name": cow_name,
                "Age": age,
                "Breed": breed,
                "Owner Name": owner_name,
                "Contact Number": phone_number,
                "Address": address
            }

            missing = [
                name for name, value in required_fields.items()
                if not value.strip()
            ]

            if missing:
                st.error(
                    "Please fill all required fields: "
                    + ", ".join(missing)
                )

            elif not cow_images:
                st.error("Please upload at least one nose image.")

            elif any(
                str(cow.get("id", "")).casefold()
                == cow_id.strip().casefold()
                for cow in cows
            ):
                st.error("This Cow ID is already registered.")

            else:

                valid, phone_result = validate_phone(
                    phone_number.strip(),
                    country_region
                )

                if not valid:
                    st.error(phone_result)

                else:
                    saved_paths = []

                    try:
                        saved_paths = save_uploaded_images(
                            cow_id,
                            cow_images
                        )

                        new_cow = {
                            "id": cow_id.strip(),
                            "name": cow_name.strip(),
                            "age": age.strip(),
                            "breed": breed.strip(),
                            "owner_name": owner_name.strip(),
                            "contact": phone_result,
                            "country": country_name,
                            "address": address.strip(),
                            "images": saved_paths,
                            "image": saved_paths[0],
                            "image_hashes": [
                                get_image_hash(
                                    open(path, "rb").read()
                                )
                                for path in saved_paths
                            ]
                        }

                        cows.append(new_cow)
                        save_cows(cows)

                        st.success(
                            f"{cow_name} registered successfully!"
                        )

                        st.success(
                            f"{len(saved_paths)} nose images saved."
                        )

                        st.balloons()

                    except Exception as error:

                        cows = load_cows()

                        for path in saved_paths:
                            if os.path.exists(path):
                                os.remove(path)

                        st.error(
                            f"Registration failed: {error}"
                        )


# =====================================================
# IDENTIFY COW
# =====================================================

with identify_tab:

    st.header("AI Cow Nose Identification")

    st.write(
        "Upload a cow nose image. AI will compare it "
        "with all registered nose images."
    )

    test_image = st.file_uploader(
        "Upload Nose Image",
        type=["jpg", "jpeg", "png"],
        key="identify_image"
    )

    if test_image:

        st.image(
            test_image,
            caption="Image to Identify",
            width=300
        )

        if st.button(
            "Identify Cow Using AI",
            type="primary"
        ):
            run_ai_identification(test_image, cows)


# =====================================================
# CAMERA SCAN – TAKE PHOTO OR LIVE AUTO SCAN
# =====================================================

with camera_tab:

    st.header("📷 Cow Nose Camera Scanner")

    scan_mode = st.radio(
        "Choose Scan Mode",
        ["Take Photo", "Live Auto Scan"],
        horizontal=True,
        key="camera_scan_mode",
    )

    if scan_mode == "Take Photo":

        st.write(
            "Capture a cow nose image. The existing manual scan flow is unchanged."
        )

        camera_image = st.camera_input(
            "Capture Cow Nose Image",
            key="take_photo_camera",
        )

        if camera_image:

            st.image(
                camera_image,
                caption="Captured Nose Image",
                width=300
            )

            if st.button(
                "Scan and Identify Cow",
                type="primary",
                key="take_photo_identify_button",
            ):
                run_ai_identification(camera_image, cows)

    else:

        st.write(
            "Start the camera and place the cow nose inside the green guide. "
            "The system will analyze frames automatically. On phones, use the "
            "camera dropdown beside the camera control to switch between available cameras."
        )

        st.caption(
            "Live mode scans the central guide area every 1.5 seconds. "
            "A match requires at least 75% similarity, a 6-point lead over "
            "the next cow, and the same candidate in 3 consecutive scans."
        )

        if not WEBRTC_AVAILABLE:
            st.error(
                "Live scanning dependency is not installed. Add streamlit-webrtc "
                "to requirements.txt and redeploy. Details: " + WEBRTC_IMPORT_ERROR
            )

        elif not AI_READY:
            st.error("AI model is not available. Check the model loading error above.")

        elif not cows:
            st.warning("Register at least one cow with nose images before live scanning.")

        else:
            with st.spinner("Preparing registered cow nose images for live scanning..."):
                live_references = prepare_live_reference_embeddings(cows)

            if not live_references:
                st.warning(
                    "No usable registered nose images were found. Check the cow_images folder."
                )
            else:
                st.caption(
                    f"Ready: {len(live_references)} registered nose-image reference(s) loaded."
                )

                if "live_scan_state" not in st.session_state:
                    st.session_state.live_scan_state = LiveScanState()
                if "live_camera_enabled" not in st.session_state:
                    st.session_state.live_camera_enabled = False

                live_state = st.session_state.live_scan_state
                video_callback = create_live_video_callback(
                    live_references,
                    live_state,
                )

                # Select the camera BEFORE opening the stream. On some Android
                # browsers, switching deviceId while a WebRTC track is already
                # active raises NotReadableError. A new component key per facing
                # mode ensures the old browser stream is not reused.
                camera_options = {
                    "Back camera (recommended)": "environment",
                    "Front camera": "user",
                }
                selected_camera_label = st.radio(
                    "Choose Live Camera",
                    list(camera_options.keys()),
                    horizontal=True,
                    key="live_camera_facing_choice",
                    disabled=st.session_state.live_camera_enabled,
                )
                selected_facing_mode = camera_options[selected_camera_label]

                if st.session_state.live_camera_enabled:
                    st.caption(
                        "To change cameras, stop Live Scan first, select the other camera, "
                        "then press Start again."
                    )
                else:
                    st.caption(
                        "Choose Back camera or Front camera before starting. "
                        "The camera is opened only after you press Start Live Scan."
                    )

                # The camera is controlled from our own Start/Stop buttons.
                # A confirmed match switches this value to False, stopping the
                # WebRTC camera; it remains off until the user starts again.
                webrtc_context = webrtc_streamer(
                    key=f"cow_nose_live_auto_scan_{selected_facing_mode}",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=get_live_rtc_configuration(),
                    # Use the browser's facingMode constraint at initial capture
                    # instead of changing an already-open camera track mid-stream.
                    media_stream_constraints={
                        "video": {"facingMode": {"exact": selected_facing_mode}},
                        "audio": False,
                    },
                    video_frame_callback=video_callback,
                    desired_playing_state=st.session_state.live_camera_enabled,
                    # Hide the in-stream device dropdown. Camera selection is
                    # handled above before starting, avoiding mobile switch errors.
                    media_toggle_controls=False,
                    async_processing=True,
                )

                current_result = live_state.snapshot()

                if current_result["status"] == "matched":
                    # Safety net in case this run observed the match before
                    # the fragment requested the app-level rerun.
                    if st.session_state.live_camera_enabled:
                        st.session_state.live_camera_enabled = False
                        st.rerun(scope="app")

                    st.success("✅ COW MATCH FOUND — CAMERA STOPPED")
                    show_match_details(
                        current_result["cow"],
                        current_result["score"],
                        current_result["matched_image"],
                    )

                    if st.button(
                        "START LIVE SCAN AGAIN",
                        type="primary",
                        key="restart_live_scan_after_match",
                    ):
                        live_state.reset("Camera starting. Place the cow nose inside the guide.")
                        st.session_state.live_camera_enabled = True
                        st.rerun(scope="app")

                elif st.session_state.live_camera_enabled:
                    if webrtc_context.state.playing:
                        st.success("Camera connected. Live auto scanning is active.")
                    else:
                        st.info("Connecting to camera... Allow camera permission if requested.")

                    if st.button("STOP LIVE SCAN", key="stop_live_scan_button"):
                        st.session_state.live_camera_enabled = False
                        live_state.reset("Live scanning stopped. Press START LIVE SCAN when ready.")
                        st.rerun(scope="app")

                    render_live_scan_result(live_state)

                else:
                    st.info("📷 Camera is OFF. Press START LIVE SCAN when you are ready.")
                    if st.button(
                        "START LIVE SCAN",
                        type="primary",
                        key="start_live_scan_button",
                    ):
                        live_state.reset("Camera starting. Place the cow nose inside the guide.")
                        st.session_state.live_camera_enabled = True
                        st.rerun(scope="app")


# =====================================================
# REGISTERED RECORDS
# =====================================================

with records_tab:

    st.header("Registered Cow Records")

    st.metric(
        "Total Registered Cows",
        len(cows)
    )

    if not cows:

        st.info("No cows registered yet.")

    else:

        for cow_index, cow in enumerate(cows):

            cow_id = str(cow.get("id", ""))
            confirm_key = f"confirm_delete_cow_{cow_index}_{cow_id}"

            with st.container(border=True):

                st.subheader(
                    cow.get("name", "Unknown Cow")
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.write("**Cow ID:**", cow.get("id", "N/A"))
                    st.write("**Age:**", cow.get("age", "N/A"))
                    st.write("**Breed:**", cow.get("breed", "N/A"))
                    st.write("**Owner:**", cow.get("owner_name", "N/A"))

                with col2:

                    st.write("**Contact:**", cow.get("contact", "N/A"))
                    st.write("**Country:**", cow.get("country", "N/A"))
                    st.write("**Address:**", cow.get("address", "N/A"))

                image_paths = get_cow_images(cow)

                st.write(
                    f"**Registered Nose Images:** {len(image_paths)}"
                )

                if image_paths:

                    image_cols = st.columns(4)

                    for image_index, path in enumerate(image_paths):

                        with image_cols[image_index % 4]:

                            st.image(
                                path,
                                caption=f"Image {image_index + 1}",
                                use_container_width=True
                            )

                st.divider()

                if st.session_state.get(confirm_key, False):

                    st.warning(
                        f"Are you sure you want to permanently delete "
                        f"{cow.get('name', 'this cow')} (ID: {cow_id})?"
                    )

                    confirm_col, cancel_col = st.columns(2)

                    with confirm_col:

                        if st.button(
                            "Yes, Delete Permanently",
                            key=f"yes_delete_{cow_index}_{cow_id}",
                            type="primary",
                            use_container_width=True
                        ):

                            # Remove only the selected record.
                            updated_cows = [
                                item for item_index, item in enumerate(cows)
                                if item_index != cow_index
                            ]

                            try:
                                save_cows(updated_cows)

                                # Delete only this cow's image files
                                # located inside the cow_images directory.
                                image_root = Path(IMAGE_FOLDER).resolve()

                                for image_path in get_cow_images(cow):
                                    try:
                                        resolved_path = Path(image_path).resolve()

                                        if (
                                            resolved_path.is_relative_to(image_root)
                                            and resolved_path.is_file()
                                        ):
                                            resolved_path.unlink()

                                    except (OSError, ValueError):
                                        pass

                                st.session_state.pop(confirm_key, None)
                                st.success(
                                    f"Cow record {cow_id} deleted successfully."
                                )
                                st.rerun()

                            except Exception as error:
                                st.error(f"Could not delete record: {error}")

                    with cancel_col:

                        if st.button(
                            "Cancel",
                            key=f"cancel_delete_{cow_index}_{cow_id}",
                            use_container_width=True
                        ):
                            st.session_state.pop(confirm_key, None)
                            st.rerun()

                else:

                    if st.button(
                        "🗑️ Delete Record",
                        key=f"delete_cow_{cow_index}_{cow_id}",
                        type="secondary"
                    ):
                        st.session_state[confirm_key] = True
                        st.rerun()

    st.divider()

    st.caption(
        "AI recognition is a prototype. Similarity scores are not "
        "proof of an animal's identity. Recognition accuracy depends "
        "on image quality, camera angle, lighting and the dataset."
    )
