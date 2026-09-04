# Adding a new software entry

Add new resources to `README.md`, under the most relevant existing section. Keep the entry to one Markdown bullet and follow the style of the surrounding examples.

## Required information

- Link the software name to its canonical GitHub repository URL. Prefer the repository over a project, documentation, package-index, or organisation page.
- Add a short, factual description after `->`. Explain what the software does and, where useful, name its main language or framework.
- If the software has an associated paper, include the paper's full, exact title and link it to the publisher, DOI, or arXiv page. Including the title makes the paper searchable with `Control+F`.
- Preserve the project's official spelling and capitalisation.

Use this form when there is no paper:

```markdown
* [ProjectName](https://github.com/owner/repository) -> A concise description of the software and its relevance to satellite or aerial imagery ML.
```

Use this form when there is a paper:

```markdown
* [ProjectName](https://github.com/owner/repository) -> A concise description. Paper: [Full Paper Title](https://doi.org/...)
```

Existing entries illustrate acceptable variations, such as:

```markdown
* [pyjeo](https://github.com/ec-jrc/jeolib-pyjeo) -> a library for image processing for geospatial data implemented in JRC Ispra, with [paper](https://www.mdpi.com/2220-9964/8/10/461)
* [Joint Learning from Earth Observation and OpenStreetMap Data to Get Faster Better Semantic Maps](https://arxiv.org/abs/1705.06057) -> fusion based architectures and coarse-to-fine segmentation to include the OpenStreetMap layer into multispectral-based deep fully convolutional networks, arxiv paper
```

For new software that has a paper, prefer the explicit `Paper: [Full Paper Title](URL)` form above so both the GitHub URL and searchable paper name are present.

## Before finishing

- Search `README.md` for the project, repository URL, and paper title to avoid duplicates.
- Confirm the entry is in the best matching section; only add a new section when none of the existing sections fits.
- Check that every URL is complete and that the Markdown renders as a single bullet.
- Keep the surrounding ordering and formatting intact. Do not reformat unrelated entries.
