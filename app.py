
import streamlit as st
import os
import json
import hashlib
import re
import phonenumbers

from PIL import Image
from io import BytesIO
from pathlib import Path

from ai_recognition import load_ai_model, compare_images


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

# Initial threshold. Calibrate with real cow images.
AI_THRESHOLD = 0.70


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


def identify_cow_ai(query_image, cows):
    """
    Compares query image against every saved image
    of every registered cow.

    Returns the best matching cow and similarity.
    """
    best_cow = None
    best_score = 0.0
    best_image = None

    query_image = query_image.convert("RGB")

    for cow in cows:
        image_paths = get_cow_images(cow)

        for image_path in image_paths:
            try:
                registered_image = Image.open(image_path).convert("RGB")

                score = ai_similarity(
                    query_image,
                    registered_image
                )

                if score > best_score:
                    best_score = score
                    best_cow = cow
                    best_image = image_path

            except Exception as error:
                print(f"Could not process {image_path}: {error}")

    if best_cow and best_score >= AI_THRESHOLD:
        return best_cow, best_score, best_image

    return None, best_score, best_image


def show_match_details(cow, score, matched_image):
    st.success("AI Matching Cow Found!")

    st.metric(
        "AI Similarity Score",
        f"{score * 100:.2f}%"
    )

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
            "The image did not reach the current AI matching threshold."
        )


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
# CAMERA SCAN
# =====================================================

with camera_tab:

    st.header("📷 AI Live Camera Scan")

    st.write(
        "Capture a cow nose image using your device camera."
    )

    camera_image = st.camera_input(
        "Capture Cow Nose Image"
    )

    if camera_image:

        st.image(
            camera_image,
            caption="Captured Nose Image",
            width=300
        )

        if st.button(
            "Scan and Identify Cow",
            type="primary"
        ):
            run_ai_identification(camera_image, cows)


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
