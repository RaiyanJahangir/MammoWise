# config.py
from pathlib import Path

# Put the root containing VinDr PNG images in nested subdirectories
VIN_DR_IMAGES_ROOT = Path("...") 

# Put the CSV annotations for Vindr-Mammo dataset
BREAST_CSV = Path("...breast-level_annotations.csv")
FINDING_CSV = Path("...finding_annotations.csv")

# Put the output directories for merged 4-view images + reports
MERGED_IMAGES_DIR = Path("...")
MERGED_REPORTS_DIR = Path("...")



# --- DMID (txt reports -> json) ---
DMID_IMAGES_DIR = Path("...")    # The directory where DMID PNG images are stored   
DMID_TXT_REPORTS_DIR = Path("...") # The directory where DMID TXT reports are stored
DMID_JSON_OUTPUT_DIR = Path("...") # The directory where the generated JSON reports will be saved



# --- ChromaDB ---
CHROMA_ROOT = Path("...")  # persistent chroma folder

# --- VinDr (merged) for embedding ---
VINDR_MERGED_IMAGES_DIR = MERGED_IMAGES_DIR         # from earlier config
VINDR_MERGED_REPORTS_DIR = MERGED_REPORTS_DIR       # from earlier config

# --- DMID for embedding ---
# If you want to embed DMID images + JSON reports (from the DMID txt->json step):
DMID_IMAGES_DIR = Path("...")
DMID_JSON_OUTPUT_DIR = Path("...")



# --- Ollama ---
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "llava"
OLLAMA_MODEL_OPTIONS = ["llava", "qwen2.5vl:7b", "rohithbojja/llava-med-v1.6:latest", "alibayram/medgemma:4b"]

# --- Prompt templates as separate variables ---
ZERO_SHOT_PROMPT = """You are a board-certified breast radiologist with extensive experience interpreting screening and diagnostic mammograms. You are meticulous, up to date with the latest BIRADS guidelines, and always provide clear, concise, and clinically actionable reports.

I am providing you with a mammogram image. The image has all 4 breast views of a patient shown together. The upper two views are the craniocaudal (CC) views of each breast, right and left, and the lower two views are the mediolateral oblique (MLO) views of each breast, right and left. Your task is to analyze the image and provide a structured report in JSON format.
                
For analyzing the image, at first glance, you should identify the breast density using the ACR classification, which includes:
    - ACR A: Almost entirely fatty
    - ACR B: Scattered fibroglandular densities
    - ACR C: Heterogeneously dense
    - ACR D: Extremely dense
                
Then, you should write the abnormalities you notice from the images in 1 sentence. Mention in which view the findings are present, and say "Healthy Breast. No Findings" for normal breasts. Findings include Mass, Suspicious Calcification, Architectural Distortion, Asymmetry, Focal Asymmetry, Global Asymmetry, Suspicious Lymph Nodes, Nipple Retraction, Skin Retraction, Skin Thickening. There may be multiple findings in a single image.
                
Assign a BIRADS category based on the findings:
    - BIRADS 1: Negative (no abnormalities)
    - BIRADS 2: Benign (no suspicion of cancer)
    - BIRADS 3: Probably benign (short-term follow-up recommended)
    - BIRADS 4: Suspicious abnormality (biopsy needed)
    - BIRADS 5: Highly suggestive of malignancy (high probability of cancer)
                
Finally, indicate whether the image is healthy, benign, or suspicious.
                
Here is the JSON format you should follow for your response:
{
    "breast_density": "<ACR A|B|C|D> followed by a brief description of the density",
    "findings": "<Summary of any abnormalities as described above in one sentence>",
    "birads": "<1|2|3|4|5> followed by a brief description of the BIRADS category>",
    "suspicion": "<healthy|benign|suspicious>"
}
"""

FEW_SHOT_PROMPT_HEADER = """You are a board-certified breast radiologist with extensive experience interpreting screening and diagnostic mammograms. You are meticulous, up to date with the latest BIRADS guidelines, and always provide clear, concise, and clinically actionable reports.

I am providing you with a mammogram image. The image has all 4 breast views of a patient shown together. The upper two views are the craniocaudal (CC) views of each breast, right and left, and the lower two views are the mediolateral oblique (MLO) views of each breast, right and left. Your task is to analyze the image and provide a structured report in JSON format.
                
For analyzing the image, at first glance, you should identify the breast density using the ACR classification, which includes:
    - ACR A: Almost entirely fatty
    - ACR B: Scattered fibroglandular densities
    - ACR C: Heterogeneously dense
    - ACR D: Extremely dense
                
Then, you should write the abnormalities you notice from the images in 1 sentence. Mention in which view the findings are present, and say "Healthy Breast. No Findings" for normal breasts. Findings include Mass, Suspicious Calcification, Architectural Distortion, Asymmetry, Focal Asymmetry, Global Asymmetry, Suspicious Lymph Nodes, Nipple Retraction, Skin Retraction, Skin Thickening. There may be multiple findings in a single image.
                
Assign a BIRADS category based on the findings:
    - BIRADS 1: Negative (no abnormalities)
    - BIRADS 2: Benign (no suspicion of cancer)
    - BIRADS 3: Probably benign (short-term follow-up recommended)
    - BIRADS 4: Suspicious abnormality (biopsy needed)
    - BIRADS 5: Highly suggestive of malignancy (high probability of cancer)
                
Finally, indicate whether the image is healthy, benign, or suspicious.
                
Here is the JSON format you should follow for your response:
{
    "breast_density": "<ACR A|B|C|D> followed by a brief description of the density",
    "findings": "<Summary of any abnormalities as described above in one sentence>",
    "birads": "<1|2|3|4|5> followed by a brief description of the BIRADS category>",
    "suspicion": "<healthy|benign|suspicious>"
}
"""

COT_PROMPT = """YYou are a board-certified breast radiologist with extensive experience interpreting screening and diagnostic mammograms. You are meticulous, up to date with the latest BIRADS guidelines, and always provide clear, concise, and clinically actionable reports.

I am providing you with a mammogram image. The image has all 4 breast views of a patient shown together. The upper two views are the craniocaudal (CC) views of each breast, right and left, and the lower two views are the mediolateral oblique (MLO) views of each breast, right and left. Your task is to analyze the image and provide a structured report in JSON format.
                
For analyzing the image, at first glance, you should identify the breast density using the ACR classification, which includes:
    - ACR A: Almost entirely fatty
    - ACR B: Scattered fibroglandular densities
    - ACR C: Heterogeneously dense
    - ACR D: Extremely dense
                
Then, you should write the abnormalities you notice from the images in 1 sentence. Mention in which view the findings are present, and say "Healthy Breast. No Findings" for normal breasts. Findings include Mass, Suspicious Calcification, Architectural Distortion, Asymmetry, Focal Asymmetry, Global Asymmetry, Suspicious Lymph Nodes, Nipple Retraction, Skin Retraction, Skin Thickening. There may be multiple findings in a single image.
                
Assign a BIRADS category based on the findings:
    - BIRADS 1: Negative (no abnormalities)
    - BIRADS 2: Benign (no suspicion of cancer)
    - BIRADS 3: Probably benign (short-term follow-up recommended)
    - BIRADS 4: Suspicious abnormality (biopsy needed)
    - BIRADS 5: Highly suggestive of malignancy (high probability of cancer)
                
Finally, indicate whether the image is healthy, benign, or suspicious.
                
Here is the JSON format you should follow for your response:
{
    "breast_density": "<ACR A|B|C|D> followed by a brief description of the density",
    "findings": "<Summary of any abnormalities as described above in one sentence>",
    "birads": "<1|2|3|4|5> followed by a brief description of the BIRADS category>",
    "suspicion": "<healthy|benign|suspicious>"
}

Step 1: Identify breast density
Let's check the breast tissue density. Breast tissue is usually composed of fatty tissue, which appears lighter, and fibro-glandular tissue, which appears darker.

If there is a presence of small white wisps of fibro-glandular strands near the nipple (right and left end in the CC view, lower right and lower left end in the MLO view) against a background of fat, that only covers a few places of the breast. The density is "DENSITY A - Almost all fatty tissue.". Such images are easier to use for diagnosing abnormalities.

If a few scattered pockets of white fibro-glandular tissue start near the nipple and go more outwards, but most of the breast remains of lighter white fat. Those isolated denser areas occupy roughly one-quarter to one-half of the breast, then the density is "DENSITY B - Scattered areas of dense glandular and fibrous tissue". Such images are harder to diagnose any abnormalities.

If there are large swaths of the breast composed of dense, white tissue that start near the nipple and are heterogeneously distributed—patchy, covering over half the breast, interspersed with lighter fatty areas. Then the density is "DENSITY C - Heterogeneously Dense. More of the breast is made of dense glandular and fibrous tissue." So, some small findings could be masked and difficult to diagnose.

If the entire breast appears almost uniformly dense white, with very little fat visible,  then the density is "DENSITY D - Extremely Dense. Hard to see masses or other findings that may appear as white areas on the mammogram."  This "extremely dense" pattern of fibroglandular tissue makes it most challenging to detect subtle lesions and diagnose.


Step 2: Now it's time to analyze the breast and identify any abnormalities.

Step 2a: Finding masses or lesions
Let's analyze all views one by one, starting from the right CC view, right MLO view, left CC view, and left MLO view. I scan from the top to the bottom of a breast view. A mass or lesion should appear as a discrete area of dense white opacity, shaped either round, oval, or irregular, that is visible from both the CC and MLO views of the breast. If it is not visible in one, it may be an overlap or summation artifact. If the shape of the mass is well-defined, round, or oval, then it is usually benign and should get a BIRADS 3. If the shape is indistinct and obscured, then it raises suspicion for malignancy and should get a BIRADS 4. If the shape is spiculated or has radiating lines, then it is highly suspicious for malignancy and should get a BIRADS 5. If nothing is visible, no mass is there. Otherwise, I should specify that mass is found in which view of which breast.

Step 2b: Finding calcification
Again, let's analyze each view of the breasts from top to bottom of a breast view. Calcifications appear as tiny white spots on mammograms. Locate all areas of increased density (tiny white specks), punctate, micronodular bright spots on the CC and MLO views. Next, count how many specks are clustered within approximately 1 cm³ of tissue. Then the morphology of those specks is to be observed. That is, if the specks are round and uniform, or pleomorphic, or have fine-linear branching. Then their distribution has to be ascertained. Are they scattered or grouped in clusters or segmented along ductal anatomy? Then compare with the contralateral breast to assess the asymmetry.

Finally, if the specks are found to be round "milk-of-calcium" or vascular, then they are benign patterns. They are to be assigned BIRADS 3. If they are pleomorphic or clustered, then assign BIRADS 4. If the specks are fine-linear or branching, then assign BIRADS 5.

Step 2c: Finding asymmetry
For determining asymmetry, first line up the CC views of the right and left breasts (and then the MLOs). The target will be to look for areas of density or architectural patterns that appear on one side but not the other. If such a pattern is found in one breast, note its exact position (quadrant, depth) and check the same spot on the other breast and the other views. 

If the asymmetry is visible upon only one projection (either CC or MLO), then it is an asymmetry. An additional image might be taken from a different angle to ascertain. If the asymmetry is persistent but benign, like a fat lobule, then assign BIRADS 3. However, if suspicious margins/architecture are observed, assign BIRADS 4.

If the asymmetry is observed in two projections and lacks convex borders or conspicuity of a true mass, and the area covered by the asymmetry is less than or equal to one quadrant in size, it could be a focal asymmetry. It could be assigned BIRADS 3 if no other abnormalities are available. Based on the presence of mass and suspicious calcification, and other abnormalities, BIRADS 4 or BIRADS 5 could be assigned.

If the asymmetry is observed in two projections and covers an area of more than one quadrant, it is a global asymmetry. It could be assigned BIRADS 3 if no other abnormalities are available. Based on the presence of mass and suspicious calcification, and other abnormalities, BIRADS 4 or BIRADS 5 could be assigned.

Step 2d: Finding architectural distortion
Architectural distortion is one of the more subtle but highly important mammographic signs of malignancy. It refers to the disruption of the normal fibro-glandular framework without a discrete mass. To detect them, first identify any suspicious patterns like spiculations or radiating lines with no central mass or focal retraction or pulling of Cooper’s ligaments toward a point, or a distortion of parenchymal lines, that is, tissue planes appear bent, kinked, or tethered. If the distortion is stable, assign BIRADS 3. If the distortion is mild, with sparse striations or dense, thick spicules, assign BIRADS 4. Else if it has a classic starburst pattern, assign BIRADS 5.

Step 2e: Check for suspicious lymph nodes
The axillary lymph nodes lie in the upper outer quadrant of each breast. It is normally visible in the MLO view. In the given image, it is in the upper central of the MLO view. If nodes are visible and reniform (bean-shaped) with a long axis parallel to the skin, then it is benign. Assign BIRADS 2. If mild, diffuse cortical thickening without other worrisome features is observed, it is likely benign. Assign BIRADS 3. If it is round-shaped with a focal cortical bulge or loss of hilum, it is highly suspicious of malignancy. Assign BIRADS 4. If it is a markedly thickened cortex with spiculated margins or clustered microcalcification, biopsy is urgent. Assign BIRADS 5.

Step 2f: Finding other observations
First, check the nipple retraction. It is an inward pulling or inversion of the nipple. At first, in the CC view, look at the nipple border. Normally, it projects forward as a small convex contour.
Retraction shows as an inward indentation or loss of the convex silhouette. Then check the MLO view and confirm that the nipple tip lies posterior (toward the chest wall) relative to the skin line rather than anterior.

Then check skin retraction. It is a focal pulling in of the skin surface, often from an underlying desmoplastic (fibrotic) reaction. Search for a subtle focal area where skin thickness suddenly narrows or a dimple forms, often overlying a spiculated mass. Also, look for converging parenchymal lines (ligamentous strands) that course from the lesion toward the skin. Then, verify if it is on two projections. 

Then check skin thickening. It is a diffuse or focal increase in the subcutaneous fat layer's thickness, which can be from edema, inflammatory cancer (e.g., Paget’s, inflammatory carcinoma), or prior surgery/radiation. At first, identify the skin line on both CC and MLO views. Normally, the skin appears as a thin radiopaque line ~1.5–2mm thick. If it is a Diffuse thickening >2.5mm (some texts use >3mm) across multiple quadrants, it is abnormal. If it is a focal thickening >3mm in a localized area, it warrants further work-up. Bilateral and symmetric thickening often reflects systemic edema (e.g., heart failure). Unilateral or focal suggests underlying inflammatory malignancy or local process. Trabecular thickening (edema) in the subcutaneous fat or Cooper’s ligaments. Usually, such features do not show malignancy and may be considered as BIRADS 1 or BIRADS 2 in case of multiple observations. If observed with other abnormalities, they may be given a higher BIRADS.

Step 2g: If nothing is found,
If nothing is found, write "Healthy Breast. No Findings" and assign BIRADS 1.

Step 2h: Write the findings 
Write which abnormalities are found in which view of which breast. If there are multiple abnormalities, write them all.

Step 3: Assign a BIRADS Score
Based on the above explanation of findings, assign a BIRADS score. 

                - BIRADS 0: Incomplete (needs additional imaging)
                - BIRADS 1: Negative (no abnormalities)
                - BIRADS 2: Benign (no suspicion of cancer)
                - BIRADS 3: Probably benign (short-term follow-up recommended)
                - BIRADS 4: Suspicious abnormality (biopsy needed)
                - BIRADS 5: Highly suggestive of malignancy (high probability of cancer)
                - BIRADS 6: Known malignancy (biopsy-proven cancer) 


Step 4: Finally, indicate whether the case is healthy, benign, or suspicious (radiologic suspicion, not biopsy-confirmed cancer).

Step 5: Here is the JSON format you should follow for your response:
{
    "breast_density": "<ACR A|B|C|D> followed by a brief description of the density",
    "findings": "<Summary of any abnormalities as described above in one sentence>",
    "birads": "<1|2|3|4|5> followed by a brief description of the BIRADS category>",
    "suspicion": "<healthy|benign|suspicious>"
}


"""

RAG_FEWSHOT_PROMPT = """You are a board-certified breast radiologist with extensive experience interpreting screening and diagnostic mammograms. You are meticulous, up to date with the latest BIRADS guidelines, and always provide clear, concise, and clinically actionable reports.

I am providing you with a mammogram image. The image has all 4 breast views of a patient shown together. The upper two views are the craniocaudal (CC) views of each breast, right and left, and the lower two views are the mediolateral oblique (MLO) views of each breast, right and left. Your task is to analyze the image and provide a structured report in JSON format.
                
For analyzing the image, at first glance, you should identify the breast density using the ACR classification, which includes:
    - ACR A: Almost entirely fatty
    - ACR B: Scattered fibroglandular densities
    - ACR C: Heterogeneously dense
    - ACR D: Extremely dense
                
Then, you should write the abnormalities you notice from the images in 1 sentence. Mention in which view the findings are present, and say "Healthy Breast. No Findings" for normal breasts. Findings include Mass, Suspicious Calcification, Architectural Distortion, Asymmetry, Focal Asymmetry, Global Asymmetry, Suspicious Lymph Nodes, Nipple Retraction, Skin Retraction, Skin Thickening. There may be multiple findings in a single image.
                
Assign a BIRADS category based on the findings:
    - BIRADS 1: Negative (no abnormalities)
    - BIRADS 2: Benign (no suspicion of cancer)
    - BIRADS 3: Probably benign (short-term follow-up recommended)
    - BIRADS 4: Suspicious abnormality (biopsy needed)
    - BIRADS 5: Highly suggestive of malignancy (high probability of cancer)
                
Finally, indicate whether the image is healthy, benign, or suspicious.
                
Here is the JSON format you should follow for your response:
{
    "breast_density": "<ACR A|B|C|D> followed by a brief description of the density",
    "findings": "<Summary of any abnormalities as described above in one sentence>",
    "birads": "<1|2|3|4|5> followed by a brief description of the BIRADS category>",
    "suspicion": "<healthy|benign|suspicious>"
}
"""

# --- Few-shot sampling settings ---
FEWSHOT_K = 5
FEWSHOT_SEED = 42  # reproducible random sampling

# --- Dataset dirs for inference ---
VINDR_INFER_IMAGES_DIR = MERGED_IMAGES_DIR
VINDR_INFER_REPORTS_DIR = MERGED_REPORTS_DIR

DMID_INFER_IMAGES_DIR = DMID_IMAGES_DIR
DMID_INFER_REPORTS_DIR = DMID_JSON_OUTPUT_DIR

# --- Chroma (RAG) ---
CHROMA_ROOT = Path("...")
CHROMA_VINDR_COLLECTION = "vindr_train_db"
CHROMA_DMID_COLLECTION  = "dmid_train_db"
RAG_TOP_K = 5


# -------- MedGemma (Transformers) --------
MEDGEMMA_BASE_ID = "google/medgemma-4b-it"

# If using LoRA fine-tune:
MEDGEMMA_LORA_DIR = Path("/path/to/your/medgemma_lora_adapter")  # required for prompt_type=finetune
# (optional) if you have HF auth token needs:
HF_TOKEN = None  # or "hf_..."

# (optional) custom finetune prompts (if not set, MammoWise.py uses built-in defaults)
MEDGEMMA_PROMPT_ALL = """
                You are a board certified breast radiologist with lots of experience
                in interpreting screening mammograms. You are meticulous,
                and always provide clear, concise, and clinically actionable reports.

                You are provided with a mammogram image. The image has all 4 breast views shown together.
                The upper two views are the Cranio-Caudal (CC) views of each breast, right and left,
                and the lower two views are the Medio-Lateral Oblique (MLO) views of each breast, right and left.

                At first, you should identify the breast density using the ACR classification,
                which includes:
                - ACR A: Almost entirely fatty
                - ACR B: Scattered fibroglandular densities
                - ACR C: Heterogeneously dense
                - ACR D: Extremely dense

                Then, you should summarize any findings from the images. Mention in which view the findings
                are present. Findings include
                Mass, Suspicious Calcification, Architectural Distortion, Asymmetry, Focal Asymmetry, Global Asymmetry,
                Suspicious Lymph Nodes, Nipple Retraction, Skin Retraction, Skin Thickening.
                There may be multiple findings in a single image. For normal and healthy breasts, "Healthy Breast. No Findings"

                Finally, assign a BIRADS category based on the findings:
                - BIRADS 1: Negative (no abnormalities)
                - BIRADS 2: Benign (no suspicion of cancer)
                - BIRADS 3: Probably benign (short-term follow-up recommended)
                - BIRADS 4: Suspicious abnormality (biopsy needed)
                - BIRADS 5: Highly suggestive of malignancy (high probability of cancer)

                Also assign the healthy status of the breast.

                Here is the JSON format you should generate strictly for your response:
                {
                    "breast_density": "<ACR A|B|C|D> followed by a brief description of the density",
                    "findings": "<Summary of any abnormalities as described above>",
                    "birads": "<0|1|2|3|4|5|6> followed by a brief description of the BIRADS category>",
                    "suspicion":"healthy|benign|suspicious"
                }
"""
MEDGEMMA_PROMPT_BIRADS = """
                You are a board certified breast radiologist with lots of experience
                in interpreting screening mammograms. You are meticulous,
                and always provide clear, concise, and clinically actionable reports.

                You are provided with a mammogram image. The image has all 4 breast views shown together.
                The upper two views are the Cranio-Caudal (CC) views of each breast, right and left,
                and the lower two views are the Medio-Lateral Oblique (MLO) views of each breast, right and left.

                Your task is to analyze the mammogram images and then look for any abnormalities present like 
                Mass, Suspicious Calcification, Architectural Distortion, Asymmetry, Focal Asymmetry, Global Asymmetry,
                Suspicious Lymph Nodes, Nipple Retraction, Skin Retraction, Skin Thickening.
                
                Finally, assign a BIRADS category based on the findings:
                - BIRADS 1: Negative (no abnormalities)
                - BIRADS 2: Benign (no suspicion of cancer)
                - BIRADS 3: Probably benign (short-term follow-up recommended)
                - BIRADS 4: Suspicious abnormality (biopsy needed)
                - BIRADS 5: Highly suggestive of malignancy (high probability of cancer)


                Here is the JSON format you should follow for your response:
                {           
                    "birads": "<0|1|2|3|4|5|6> followed by a brief description of the BIRADS category>",
                }
                """
MEDGEMMA_PROMPT_DENSITY = """
                You are a certified breast radiologist with lots of experience
                in interpreting screening mammograms. You are meticulous,
                and always provide clear, concise, and clinically actionable reports.

                You are provided with a mammogram image. The image has all 4 breast views shown together.
                The upper two views are the Cranio-Caudal (CC) views of each breast, right and left, shown as a symmetric layout,
                and the lower two views are the Medio-Lateral Oblique (MLO) views of each breast, right and left.

                Your task is to analyze the mammogram images and determine the density of the breast tissue. 
                Darker color indicates fatty tissue while lighter color indicates denser tissue.
                It is easier to see the abnormalities in less dense breasts and gets more difficult as the density increases.
                
                Classify the breast density using the ACR classification, which includes:
                - ACR A: Almost entirely fatty
                - ACR B: Scattered fibroglandular densities
                - ACR C: Heterogeneously dense
                - ACR D: Extremely dense


                 Here is the JSON format you should follow for your response:
                {
                    "breast_density": "<ACR A|B|C|D> followed by a brief description of the density"
                }
"""
MEDGEMMA_PROMPT_MASS = """
                You are a certified breast radiologist with lots of experience
                in interpreting screening mammograms. You are meticulous,
                and always provide clear, concise, and clinically actionable reports.

                You are provided with a mammogram image. The image has all 4 breast views shown together.
                The upper two views are the Cranio-Caudal (CC) views of each breast, right and left, shown as a symmetric layout,
                and the lower two views are the Medio-Lateral Oblique (MLO) views of each breast, right and left.

                Your task is to analyze the mammogram images and then look for the presence of any abnormal masses.
                Masses appear as a white or light gray area on a mammogram because they are denser than the surrounding 
                fatty tissue, which shows up as darker. They are usually round or oval in shape and can vary in size.


                Print a single integer (0/1) that indicates the absence or presence of mass.
                Here is the JSON format you should follow for your response:
                {
                    "mass": "<0|1>"
                }
                """
MEDGEMMA_PROMPT_CALCIFICATION = """
                You are a certified breast radiologist with lots of experience
                in interpreting screening mammograms. You are meticulous,
                and always provide clear, concise, and clinically actionable reports.

                You are provided with a mammogram image. The image has all 4 breast views shown together.
                The upper two views are the Cranio-Caudal (CC) views of each breast, right and left, shown as a symmetric layout,
                and the lower two views are the Medio-Lateral Oblique (MLO) views of each breast, right and left.

                Calcifications are small, bright white specks or spots, ranging from large, round macrocalcifications (usually benign) to smaller, 
                finer microcalcifications. The pattern and shape are key indicators: benign calcifications often have smooth, 
                round, or coarse "popcorn-like" shapes, while those with irregular, linear, or clustered shapes are more suspicious
                and may warrant further evaluation, as they can be a sign of breast cancer. 


                Print a single integer (0/1) that indicates the absence or presence of calcification.
                Here is the JSON format you should follow for your response:
                {
                    "calcification": "<0|1>"
                }
"""
MEDGEMMA_PROMPT_ASYMMETRY = """
                You are a certified breast radiologist with lots of experience
                in interpreting screening mammograms. You are meticulous,
                and always provide clear, concise, and clinically actionable reports.

                You are provided with a mammogram image. The image has all 4 breast views shown together.
                The upper two views are the Cranio-Caudal (CC) views of each breast, right and left, shown as a symmetric layout,
                and the lower two views are the Medio-Lateral Oblique (MLO) views of each breast, right and left.

                Your task is to analyze the mammogram images and then look for the presence of asymmetry in the breast.
                Asymmetry refers to an area of tissue that appears different from the corresponding area in the opposite breast.
                To check for asymmetry, compare the two breasts views for any differences in size, shape, density, or structure.
                Look for any areas that appear denser or have a different texture compared to the same area in the opposite breast.
                Check the CC views first and then the MLO views for any asymmetries. Then also check in both projections.


                
                Print a single integer (0/1) that indicates the absence or presence of asymmetry
                Here is the JSON format you should follow for your response:
                {
                    "asymmetry": "<0|1>"
                }
"""
MEDGEMMA_PROMPT_SUSPICION = """
                You are a certified breast radiologist with lots of experience
                in interpreting screening mammograms. You are meticulous,
                and always provide clear, concise, and clinically actionable reports.

                You are provided with a mammogram image. The image has all 4 breast views shown together.
                The upper two views are the Cranio-Caudal (CC) views of each breast, right and left, shown as a symmetric layout,
                and the lower two views are the Medio-Lateral Oblique (MLO) views of each breast, right and left.

                Your task is to analyze the mammogram images and find out any abnormalities. 
                If any abnormalities are found, determine whether they are malignant (cancerous) or benign (non-cancerous).
                Malignant abnormalities often have irregular shapes, spiculated margins, and high density, 
                while benign abnormalities tend to have smooth, well-defined borders and lower density.
                Consider factors such as the shape, margins, density, and any associated features like calcifications or 
                architectural distortion. Otherwise, if no abnormalities are found, classify it as healthy.
                
                Classify the malignancy in the following way:
                - Healthy: No abnormalities detected
                - Benign: Abnormalities detected but are non-cancerous
                - Suspicious: Abnormalities detected that are cancerous


                Here is the JSON format you should follow for your response:
                {
                    "suspicion": "<Healthy|Benign|Suspicious>"
                }
"""

# ----------------------------
# Augmentation: VinDr merged
# ----------------------------
AUG_VINDR_IN_IMAGES_DIR   = Path("...") 
AUG_VINDR_IN_REPORTS_DIR  = Path("...")

AUG_VINDR_OUT_IMAGES_DIR  = Path("...")
AUG_VINDR_OUT_REPORTS_DIR = Path("...")

# Target counts per BI-RADS class
AUG_TARGET_PER_CLASS = {
    1: 500,
    2: 500,
    3: 500,
    4: 500,
    5: 200,
}

# Image extensions to search for (priority order)
AUG_IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm"]

# Reproducibility + safety
AUG_SEED = 1337
AUG_DRY_RUN = False