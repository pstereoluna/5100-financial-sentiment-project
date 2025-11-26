# How to Create a Presentation

This guide shows you multiple ways to create a presentation from your project results.

## Option 1: Export Jupyter Notebook as Slides (Recommended)

### Step 1: Configure Notebook for Slides

1. Open your notebook: `notebooks/03_full_pipeline_report.ipynb`
2. In Jupyter, go to **View → Cell Toolbar → Slideshow**
3. Set each cell's slide type:
   - **Slide**: Main content slides
   - **Sub-slide**: Supporting points
   - **Fragment**: Appear on click
   - **Skip**: Hidden from presentation
   - **Notes**: Speaker notes

### Step 2: Recommended Slide Structure

```
Cell 1: Title Slide (Slide)
Cell 2: Problem Statement (Slide)
Cell 3: Dataset Overview (Slide)
Cell 4: Methodology (Slide)
Cell 5: Results - Performance (Slide)
Cell 6: Results - Visualizations (Slide)
Cell 7: Label Quality Analysis (Slide)
Cell 8: Key Findings (Slide)
Cell 9: Conclusion (Slide)
```

### Step 3: Export to Slides

**Method A: Using Jupyter**
```bash
# Install reveal.js exporter
pip install jupyter-reveal

# Export notebook to reveal.js slides
jupyter nbconvert notebooks/03_full_pipeline_report.ipynb --to slides
```

**Method B: Using nbconvert**
```bash
# Export to HTML slides
jupyter nbconvert notebooks/03_full_pipeline_report.ipynb \
    --to slides \
    --reveal-prefix reveal.js \
    --output presentation.html
```

**Method C: Export to PDF**
```bash
# Export to PDF (via HTML)
jupyter nbconvert notebooks/03_full_pipeline_report.ipynb \
    --to pdf \
    --output presentation.pdf
```

---

## Option 2: Use the Presentation Markdown

1. **Open**: `PRESENTATION.md`
2. **Convert to slides** using:
   - **Marp** (VS Code extension)
   - **reveal.js**
   - **Pandoc** to PowerPoint
   - **Google Slides** (copy content)

### Using Marp (VS Code)

1. Install Marp extension in VS Code
2. Open `PRESENTATION.md`
3. Click "Marp: Export Slide Deck"
4. Choose format (PDF, HTML, PPTX)

### Using Pandoc

```bash
# Install pandoc
brew install pandoc  # Mac
# or download from https://pandoc.org

# Convert to PowerPoint
pandoc PRESENTATION.md -o presentation.pptx

# Convert to PDF (requires LaTeX)
pandoc PRESENTATION.md -o presentation.pdf
```

---

## Option 3: Create PowerPoint from Results

### Automated Script

```bash
# Generate summary first
python create_presentation.py

# Then manually create PowerPoint using:
# - PRESENTATION_SUMMARY.md for content
# - results/*.png for visualizations
```

### Manual Steps

1. **Open PowerPoint/Google Slides**
2. **Use `PRESENTATION.md`** as content outline
3. **Insert visualizations** from `results/`:
   - `confusion_matrix.png`
   - `label_distribution.png`
   - `top_features.png`
   - `label_quality_analysis.png`
4. **Add key metrics** from `PRESENTATION_SUMMARY.md`

---

## Option 4: Interactive Presentation with Jupyter

### Present Directly from Notebook

1. **Install RISE** (Reveal.js Jupyter extension):
```bash
pip install rise
jupyter-nbextension install rise --py --sys-prefix
jupyter-nbextension enable rise --py --sys-prefix
```

2. **Open notebook** in Jupyter
3. **Click "Enter/Exit RISE Slideshow"** button
4. **Present directly** from notebook!

**Navigation**:
- Space/Arrow keys: Navigate slides
- Esc: Exit slideshow mode

---

## Quick Presentation Checklist

### Content to Include

- [ ] **Title slide** with project name
- [ ] **Problem statement** - Why this matters
- [ ] **Dataset overview** - Size, distribution
- [ ] **Methodology** - Preprocessing + Model
- [ ] **Results** - Accuracy, confusion matrix
- [ ] **Visualizations** - All key plots
- [ ] **Key findings** - Main insights
- [ ] **Future work** - Next steps
- [ ] **Q&A slide**

### Visualizations to Include

- [ ] Confusion matrix
- [ ] Label distribution
- [ ] Top features by class
- [ ] Label quality analysis
- [ ] Sample predictions (if relevant)

### Metrics to Highlight

- [ ] Overall accuracy
- [ ] Per-class precision/recall/F1
- [ ] Misclassification rate
- [ ] Ambiguous prediction rate

---

## Presentation Tips

### 1. Start Strong
- Clear problem statement
- Real-world relevance
- Impact/importance

### 2. Keep It Visual
- Use plots from `results/`
- Show examples, not just numbers
- One key point per slide

### 3. Tell a Story
- Problem → Solution → Results → Impact
- Connect technical details to business value

### 4. Practice Timing
- Aim for 10-15 minutes
- 1-2 minutes per slide
- Leave time for Q&A

### 5. Prepare for Questions
- Know your model's limitations
- Understand the data quality issues
- Have examples ready

---

## Resources

### Files Available

- `PRESENTATION.md` - Slide content outline
- `PRESENTATION_SUMMARY.md` - Auto-generated summary (run `create_presentation.py`)
- `results/*.png` - All visualizations
- `results/*.csv` - Detailed results data
- `notebooks/03_full_pipeline_report.ipynb` - Complete analysis

### Quick Commands

```bash
# Generate presentation summary
python create_presentation.py

# Export notebook to slides
jupyter nbconvert notebooks/03_full_pipeline_report.ipynb --to slides

# View results
open results/confusion_matrix.png  # Mac
xdg-open results/confusion_matrix.png  # Linux
```

---

## Example Presentation Flow

1. **Introduction** (2 min)
   - Financial sentiment analysis importance
   - Project goal

2. **Dataset & Methodology** (3 min)
   - SEntFiN dataset overview
   - Preprocessing steps
   - Model architecture

3. **Results** (5 min)
   - Performance metrics
   - Confusion matrix
   - Top features
   - Label quality insights

4. **Discussion** (3 min)
   - Key findings
   - Challenges & solutions
   - Future work

5. **Q&A** (2 min)

**Total: ~15 minutes**

---

Good luck with your presentation! 🎤



