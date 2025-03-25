# ImageFilteringHE

ImageFilteringHE is a project that demonstrates the application of homomorphic encryption techniques to perform image filtering operations on encrypted images. By leveraging the TFHE (Fast Fully Homomorphic Encryption over the Torus) library, this project enables computations on encrypted data without the need for decryption, ensuring data privacy throughout the processing pipeline.

## Objective

The primary goal of this project is to showcase how homomorphic encryption can be utilized to perform image processing tasks directly on encrypted images. This approach ensures that sensitive image data remains confidential, as the processing does not require exposing the unencrypted content. The project includes implementations of basic logical operations such as AND and OR on encrypted images, demonstrating the feasibility of encrypted domain image filtering.

## Technologies Used

- **Rust**: The project is implemented in Rust, a systems programming language known for its performance and safety features.
- **TFHE-rs**: This Rust library provides tools for implementing fully homomorphic encryption schemes, allowing computations on encrypted data.
- **Image-rs**: A Rust library used for image processing tasks, facilitating the manipulation and handling of image data within the project.

## License

This project is licensed under the Apache License 2.0. For more details, please refer to the [LICENSE](https://github.com/Edoardo-Manenti/ImageFilteringHE/blob/main/LICENSE) file in the repository.

---

*Note: This README provides a high-level overview of the project's objectives and the technologies employed. For detailed implementation insights and code examples, please refer to the source code files within the repository.*

