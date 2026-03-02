# Evaluation Data

This folder contains evaluation files prepared for TidyLang2026 Challenge.

Download Evaluation data: https://datacollective.mozillafoundation.org/datasets/cmkv32i5e02tumg07j79d3c35


## Files

- `tl26_lid.txt`
  - file list for language-ID style inference.
  - Contains only file names.

- `tl26_enroll.tsv`
  - Enrollment/reference data.
  - One row per enrollment identity.
- `tl26_pairs.txt`
  - Evaluation pairs for verification-style scoring.
  - Contains only:
    - enrollmentID
    - test file


## `tl26_enroll.tsv` Structure

Tab-separated format, no header.


Example:

`LID_8362W5S    03k2jotoge0.wav    04q8lawk8g6.wav    ...    0pklkrfw7ws.wav`

Interpretation:

- Column 1 is the enrollment identity key.
- Columns 2-9 are enrollment utterances belonging to that identity.
- Use column 1 to match against the first column in `tl26_pairs.txt`.



### Closed-Condition (Required)

Submit to the **TidyLang Closed-Condition** task with these files:

```
submission_closed.zip
├── tl26_closed_lid.txt      # Language identification output
└── tl26_closed_pairs.txt    # Verification scores
```

- Only Tidy-X training data allowed


### Open-Condition (Optional)

Submit to the **TidyLang Open-Condition** task with these files:

```
submission_closed+open.zip
├── tl26_closed_lid.txt      # Language identification output closed
├── tl26_closed_pairs.txt    # Verification scores closed
├── tl26_open_lid.txt      # Language identification output
└── tl26_open_pairs.txt    # Verification scores
```

- Additional LID datasets allowed (no extra Common Voice)

## File Formats

### Language Identification (*_lid.txt)

Each line contains a **language ID** same as training data:

**Example:**
```
en
fr
fa
de
es
```

| Requirement | Description |
|-------------|-------------|
| **One language ID per line** | Language code (e.g., "en", "fr", "fa") |
| **Exact line count** | Must match the trial list exactly |
| **No header** | Start directly with predictions |
| **UTF-8 encoding** | Use UTF-8 for language labels |

### Verification Pairs (*_pairs.txt)

Each line contains a **similarity score**:

**Example:**
```
0.862541
0.124893
0.745123
0.089234
0.956721
```

| Requirement | Description |
|-------------|-------------|
| **Score format** | Floating point number (e.g., `0.862541` or `-1.234`) |
| **One score per line** | No additional columns, tabs, or spaces |
| **Exact line count** | Must match the trial list exactly |
| **No header** | Start directly with scores |

⚠️ **Important:** Higher scores indicate higher likelihood that the enrollment and test utterance are from the same language.

## Validation Checks

The scoring system performs the following validation:

1. ✅ ZIP file contains the correct files for the selected track
2. ✅ Language ID file contains valid language labels
3. ✅ Pairs file contains valid numeric scores
4. ✅ Line counts match the expected trial counts

If any validation fails, your submission will receive an error with details about what needs to be fixed.

## Submission Limits

- **5 submissions per day**
- **20 total submissions**

## Evaluation Metrics

| File | Metrics | Better is |
|------|---------|-----------|  
| tl26_lid.txt | **Macro Accuracy (%)**, **Micro Accuracy (%)** | Higher |
| tl26_pairs.txt | **EER (%)**, **minDCF** | Lower |

## Technical Support

For technical issues during submission, contact: **aref.farhadipour@uzh.ch**
