![logo](assets/nori.png)

# Kidney Tissue Characterization using Normalized Raman Imaging (NoRI) and Segment-Anything

## Abstract
This study addresses the computational challenges of accurately segmenting and quantifying cellular components in kidney tissue using high-resolution imaging technologies like normalized Raman imaging (NoRI). To overcome the complexity of analyzing large, multi-channel datasets, we developed a computational pipeline integrating classical image processing techniques with machine learning models. The pipeline includes preprocessing, segmentation with the Segment Anything Model (SAM), pixel-based classification via Ilastik, and post-processing refinements. Key structures—tubules, nuclei, brush borders, and lumens—are segmented to quantify protein and lipid concentrations. Our approach achieves high accuracy, with an F1 score of 0.9340 for kidney tubule segmentation, enhanced by a custom contrast-enhancement technique for SAM. Comparative evaluations confirm its superiority over conventional segmentation methods, while biological analyses reveal insights into protein and lipid distribution across different kidney regions, sexes, and experimental conditions. This robust and scalable framework significantly improves segmentation accuracy and efficiency, making it broadly applicable to bioimage analysis in kidney pathology and other tissue types.

[Computation Paper](#)   |   [Bio Paper](#)   |   [Dataset](#)

---

## Contact
For questions and collaboration opportunities, contact:
* Harvard Medical School Image Analysis Collaboratory
* 📧 Email: [iac@hms.harvard.edu](mailto:iac@hms.harvard.edu)
* 🌐 Website: [HMS-IAC](https://iac.hms.harvard.edu/)
