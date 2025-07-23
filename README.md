Rapid detection of COVID-19 is essential to prevent the disease from spreading. Currently,
numerous machine learning algorithms have been developed to detect COVID-19 using
Computerized Tomography (CT) lung scans. However, due to how broad and general they
are, there is a lack of precision and attention to these patients. In particular, these
algorithms prioritize accurate detection on an image-by-image basis, instead of on a patientby-patient basis. Treating each scan independently(image-by-image) might result in a
misdiagnosis if there are multiple CT scans of a single patient and they are not all
incorporated in the final decision process. Having repeated images in different parts of the
model will produce an invalid outcome that can’t be trusted for real world scenarios.
Moreover, these developed algorithms use a single dataset, which raises concerns about
the generalization of the methods to other data. Various datasets tend to vary in image size
and quality due to differing CT machine environments. Our approach of tackling both of
these issues is to create a convolutional neural network (CNN) machine learning algorithm
that prioritizes producing an accurate diagnosis from multiple scans of a single patient.
These methodologies include (1) a voting system based on individual image predictions, and
(2) a CNN that takes multiple images from the same patient. The approach is tested with the
two largest datasets that are currently available in patient-based split.A cross dataset study
is presented to show the robustness of the models in a realistic scenario in which data
comes from different distributions.
