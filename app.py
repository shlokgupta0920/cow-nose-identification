
import streamlit as st
import os
import json
import hashlib
import io
import cv2
import numpy as np
from PIL import Image, ImageOps

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
            return json.load(file)
    return []


def save_cows(cows):
    with open(DATA_FILE, "w") as file:
        json.dump(cows, file, indent=4)


def get_image_hash(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()


# ---------------- IMAGE PROCESSING ----------------

def image_to_gray(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")

    image_array = np.array(image)

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    return gray


def compare_images(query_bytes, registered_bytes):

    query_gray = image_to_gray(query_bytes)
    registered_gray = image_to_gray(registered_bytes)

    orb = cv2.ORB_create(nfeatures=1500)

    keypoints1, descriptors1 = orb.detectAndCompute(
        query_gray, None
    )

    keypoints2, descriptors2 = orb.detectAndCompute(
        registered_gray, None
    )

    if descriptors1 is None or descriptors2 is None:
        return 0

    if len(descriptors1) < 2 or len(descriptors2) < 2:
        return 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    good_matches = []

    for pair in matches:
        if len(pair) == 2:
            first, second = pair

            if first.distance < 0.75 * second.distance:
                good_matches.append(first)

    return len(good_matches)


def find_matching_cow(query_bytes, cows):

    best_cow = None
    best_score = 0

    for cow in cows:

        image_path = cow.get("image")

        if not image_path or not os.path.exists(image_path):
            continue

        try:
            with open(image_path, "rb") as file:
                registered_bytes = file.read()

            score = compare_images(
                query_bytes,
                registered_bytes
            )

            if score > best_score:
                best_score = score
                best_cow = cow

        except Exception as error:
            st.warning(
                f"Could not process image for {cow.get('id')}: {error}"
            )

    # Initial prototype threshold; needs testing with real samples.
    if best_score >= 8:
        return best_cow, best_score

    return None, best_score


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

    cow_id = st.text_input("Cow ID")

    cow_name = st.text_input("Cow Name")

    cow_image = st.file_uploader(
        "Upload Cow Nose Image",
        type=["jpg", "jpeg", "png"],
        key="register_image"
    )

    if cow_image:
        st.image(
            cow_image,
            caption="Uploaded Nose Image",
            width=300
        )

    if st.button("Register Cow", type="primary"):

        if not cow_id or not cow_name or not cow_image:

            st.warning(
                "Please fill all details and upload an image."
            )

        elif any(c["id"] == cow_id for c in cows):

            st.error("This Cow ID is already registered.")

        else:

            image_bytes = cow_image.getvalue()
            image_hash = get_image_hash(image_bytes)

            if any(
                c.get("image_hash") == image_hash
                for c in cows
            ):

                st.error("This exact image is already registered.")

            else:

                image_path = os.path.join(
                    IMAGE_FOLDER,
                    f"{cow_id}.jpg"
                )

                with open(image_path, "wb") as file:
                    file.write(image_bytes)

                cows.append({
                    "id": cow_id,
                    "name": cow_name,
                    "image": image_path,
                    "image_hash": image_hash
                })

                save_cows(cows)

                st.success(
                    f"Cow {cow_name} registered successfully!"
                )


# ---------------- UPLOAD IDENTIFICATION ----------------

with identify_tab:

    st.header("Identify Cow Using Image Upload")

    test_image = st.file_uploader(
        "Choose Cow Nose Image",
        type=["jpg", "jpeg", "png"],
        key="identify_image"
    )

    if test_image:

        st.image(
            test_image,
            caption="Image to Identify",
            width=300
        )

        if st.button("Identify Cow", type="primary"):

            if not cows:

                st.warning("No registered cows available.")

            else:

                with st.spinner("Comparing images..."):

                    match, score = find_matching_cow(
                        test_image.getvalue(),
                        cows
                    )

                if match:

                    st.success("Matching Cow Found!")

                    st.subheader("Cow Details")

                    st.write("**Cow ID:**", match["id"])
                    st.write("**Cow Name:**", match["name"])
                    st.write("**Feature Matches:**", score)

                    if os.path.exists(match["image"]):
                        st.image(
                            match["image"],
                            caption="Registered Cow Image",
                            width=300
                        )

                else:

                    st.error("No Matching Cow Found.")

                    st.write(
                        "Best feature match score:",
                        score
                    )


# ---------------- CAMERA SCAN ----------------

with camera_tab:

    st.header("📷 Scan Cow Nose Using Camera")

    st.write(
        "Use your phone or laptop camera to scan a printed cow nose image."
    )

    camera_image = st.camera_input(
        "Open Camera and Capture Cow Nose"
    )

    if camera_image:

        st.image(
            camera_image,
            caption="Captured Cow Nose Image",
            width=400
        )

        if st.button("Scan and Identify Cow", type="primary"):

            if not cows:

                st.warning("No registered cows available.")

            else:

                with st.spinner(
                    "AI image matching in progress..."
                ):

                    match, score = find_matching_cow(
                        camera_image.getvalue(),
                        cows
                    )

                if match:

                    st.success("🐄 MATCH FOUND!")

                    st.subheader("Identified Cow Details")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Cow ID:**", match["id"])
                        st.write("**Cow Name:**", match["name"])
                        st.write("**Feature Matches:**", score)

                    with col2:

                        if os.path.exists(match["image"]):

                            st.image(
                                match["image"],
                                caption="Registered Cow",
                                width=250
                            )

                else:

                    st.error("No Matching Cow Found.")

                    st.info(
                        "Try capturing a clearer, well-lit image "
                        "with the complete nose pattern visible."
                    )

                    st.write("Best feature match score:", score)


# ---------------- REGISTERED RECORDS ----------------

with records_tab:

    st.header("Registered Cow Records")

    if not cows:

        st.info("No cows registered yet.")

    else:

        st.write(
            f"Total Registered Cows: {len(cows)}"
        )

        for cow in cows:

            with st.container(border=True):

                st.subheader(cow["name"])

                st.write("**Cow ID:**", cow["id"])

                if os.path.exists(cow["image"]):

                    st.image(
                        cow["image"],
                        width=200
                    )
