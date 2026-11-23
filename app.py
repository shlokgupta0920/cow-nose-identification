
import streamlit as st
import os
import json
import hashlib
import re
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO

# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="Cow Nose Identification System",
    page_icon="🐄",
    layout="wide"
)

DATA_FILE = "cows.json"
IMAGE_FOLDER = "cow_images"

os.makedirs(IMAGE_FOLDER, exist_ok=True)


# ---------------- DATABASE FUNCTIONS ----------------

def load_cows():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as file:
            cows = json.load(file)

        # Older records ke missing fields
        for cow in cows:
            cow.setdefault("age", "")
            cow.setdefault("owner", "")
            cow.setdefault("contact", "")
            cow.setdefault("address", "")
            cow.setdefault("breed", "")

        return cows

    return []


def save_cows(cows):
    with open(DATA_FILE, "w") as file:
        json.dump(cows, file, indent=4)


# ---------------- IMAGE MATCHING ----------------

def get_phash(image_bytes):
    """
    Perceptual hash:
    Similar-looking images ko compare karne ke liye.
    """

    image = Image.open(BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("L")
    image = image.resize((32, 32))

    pixels = np.asarray(image, dtype=np.float64)

    # DCT calculation
    n = 32
    x = np.arange(n)
    dct_matrix = np.cos(
        np.pi * (2 * x[:, None] + 1) * x[None, :] / (2 * n)
    )

    dct_matrix[:, 0] *= 1 / np.sqrt(2)
    dct_matrix *= np.sqrt(2 / n)

    dct = dct_matrix @ pixels @ dct_matrix.T

    low_freq = dct[:8, :8]
    median = np.median(low_freq.flatten()[1:])

    bits = low_freq > median

    return "".join("1" if bit else "0" for bit in bits.flatten())


def hash_distance(hash1, hash2):
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return 999

    return sum(a != b for a, b in zip(hash1, hash2))


def find_matching_cow(image_bytes, cows, threshold=10):
    try:
        test_hash = get_phash(image_bytes)
    except Exception:
        return None

    best_match = None
    best_distance = 999

    for cow in cows:
        stored_hash = cow.get("phash", "")

        if not stored_hash:
            continue

        distance = hash_distance(test_hash, stored_hash)

        if distance < best_distance:
            best_distance = distance
            best_match = cow

    if best_distance <= threshold:
        return best_match

    return None


# ---------------- DISPLAY COW DETAILS ----------------

def display_cow_details(cow):
    st.subheader("🐄 Cow Details")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Cow ID:**", cow["id"])
        st.write("**Cow Name:**", cow["name"])
        st.write("**Age:**", cow.get("age", ""))
        st.write("**Breed:**", cow.get("breed", ""))

    with col2:
        st.write("**Owner Name:**", cow.get("owner", ""))
        st.write("**Contact Number:**", cow.get("contact", ""))
        st.write("**Address:**", cow.get("address", ""))

    if os.path.exists(cow["image"]):
        st.image(cow["image"], width=300)


# ---------------- APP DESIGN ----------------

st.title("🐄 Cow Nose Identification System")

st.caption(
    "Contactless Biometric Identification and Traceability System"
)

st.divider()

cows = load_cows()

register_tab, identify_tab, camera_tab, records_tab = st.tabs(
    [
        "Register Cow",
        "Identify Cow",
        "Camera Scan",
        "Registered Records"
    ]
)


# ---------------- REGISTER COW ----------------

with register_tab:

    st.header("Register New Cow")

    with st.form("register_form"):

        col1, col2 = st.columns(2)

        with col1:
            cow_id = st.text_input("Cow ID")
            cow_name = st.text_input("Cow Name")
            age = st.text_input("Age (e.g. 3 years)")
            breed = st.text_input("Breed")

        with col2:
            owner = st.text_input("Owner Name")
            contact = st.text_input("Contact Number")
            address = st.text_area("Address")

        cow_image = st.file_uploader(
            "Upload Cow Nose Image",
            type=["jpg", "jpeg", "png"],
            key="register_image"
        )

        submitted = st.form_submit_button(
            "Register Cow",
            type="primary"
        )

        if submitted:

            cow_id = cow_id.strip()
            cow_name = cow_name.strip()

            if not all([
                cow_id, cow_name, age.strip(),
                breed.strip(), owner.strip(),
                contact.strip(), address.strip(), cow_image
            ]):
                st.warning("Please fill all details and upload an image.")

            elif any(c["id"] == cow_id for c in cows):
                st.error("This Cow ID is already registered.")

            elif not contact.strip().isdigit():
                st.error("Enter a valid contact number.")

            else:
                image_bytes = cow_image.getvalue()

                try:
                    image_hash = get_phash(image_bytes)

                    if any(
                        c.get("phash") == image_hash for c in cows
                    ):
                        st.error("This image is already registered.")

                    else:
                        safe_id = re.sub(
                            r"[^a-zA-Z0-9_-]", "_", cow_id
                        )

                        image_path = os.path.join(
                            IMAGE_FOLDER,
                            f"{safe_id}.jpg"
                        )

                        image = Image.open(BytesIO(image_bytes))
                        image = ImageOps.exif_transpose(image).convert("RGB")
                        image.save(image_path, format="JPEG")

                        new_cow = {
                            "id": cow_id,
                            "name": cow_name,
                            "age": age.strip(),
                            "owner": owner.strip(),
                            "contact": contact.strip(),
                            "address": address.strip(),
                            "breed": breed.strip(),
                            "image": image_path,
                            "phash": image_hash
                        }

                        cows.append(new_cow)
                        save_cows(cows)

                        st.success(
                            f"Cow {cow_name} registered successfully!"
                        )

                except Exception as e:
                    st.error(f"Registration error: {e}")

    if cow_image:
        st.image(cow_image, caption="Uploaded Cow Nose Image", width=300)


# ---------------- IDENTIFY COW ----------------

with identify_tab:

    st.header("Identify Cow")

    st.write("Upload a cow nose image to find its registered details.")

    test_image = st.file_uploader(
        "Choose Nose Image",
        type=["jpg", "jpeg", "png"],
        key="identify_image"
    )

    if test_image:

        st.image(test_image, caption="Image to Identify", width=300)

        if st.button("Identify Cow", type="primary"):

            match = find_matching_cow(
                test_image.getvalue(), cows
            )

            if match:
                st.success("Matching Cow Found!")
                display_cow_details(match)

            else:
                st.error("No Matching Cow Found.")


# ---------------- CAMERA SCAN ----------------

with camera_tab:

    st.header("📷 Live Camera Scan")

    st.write(
        "Phone camera se registered cow ki printed nose image scan karein."
    )

    camera_image = st.camera_input("Open Camera and Scan Cow Nose")

    if camera_image:

        image_bytes = camera_image.getvalue()

        st.image(
            image_bytes,
            caption="Captured Image",
            width=350
        )

        if st.button("Scan and Identify Cow", type="primary"):

            match = find_matching_cow(image_bytes, cows)

            if match:
                st.success("Matching Cow Found!")
                display_cow_details(match)

            else:
                st.error("No Matching Cow Found.")


# ---------------- REGISTERED RECORDS ----------------

with records_tab:

    st.header("Registered Cow Records")

    st.write(f"**Total Registered Cows: {len(cows)}**")

    if not cows:

        st.info("No cows registered yet.")

    else:

        for cow in cows:

            with st.container(border=True):

                st.subheader(cow["name"])

                display_cow_details(cow)

