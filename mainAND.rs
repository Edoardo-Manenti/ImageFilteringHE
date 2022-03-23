extern crate image;

use std::env;
use concrete::*;
use image::*;

fn main() {
	let _args: Vec<String> = env::args().collect();
	let filename1 = "mickey.png";
	let filename2 = "mickey_inv.png";
	
	// Secret keys
	let sk_rlwe = RLWESecretKey::new(&RLWE128_1024_1);
	let sk = LWESecretKey::new(&LWE128_630);
	// Bootstrappin key
	println!("Creating bootstrapping key");
	let bsk = LWEBSK::new(&sk, &sk_rlwe, 5, 3);
	bsk.save("my_bootstrapping_key.json");

	let encoder = Encoder::new(0., 255., 3, 2).expect("Can't instantiate encoder");
	let mut img_buff_1 = read_image(String::from(filename1)).expect("Can't read image");
	let img_buff_2 = read_image(String::from(filename2)).expect("Can't read image");
	let (w, h) = img_buff_1.dimensions();
	let mut c1 = encrypt_image(img_buff_1, &sk, &encoder).expect("Cant' encrpyt image");
	let c2 = encrypt_image(img_buff_2, &sk, &encoder).expect("Can't encrypt image");
	//bitwise_or(&mut c1, &c2).expect("Couldn't executo homomorphic operation");
	let result = mul_ciphertext(&c1, &c2, &bsk).expect("Can't perform AND operation");
	//let data1: Vec<f64> = decrypt_cipher_text(c1, &sk).expect("");
	//let data2: Vec<f64> = decrypt_cipher_text(c2, &sk).expect("");
	let mult_data: Vec<f64> = decrypt_cipher_text(result, &sk).expect("");
	//img_buff_1 = prepare_image(data1, &(w,h));
	//img_buff_2 = prepare_image(data2, &(w,h));
	let img_buff_mult = prepare_image(mult_data, &(w,h));
	write_image(img_buff_mult, String::from("mickey_AND_fhe.png")).expect("Can't write image");
	//write_image(img_buff_2, String::from("mickey_inv_fhe.png")).expect("Can't write image");

}

fn mul_ciphertext(c1: &VectorLWE, c2: &VectorLWE, bsk: &LWEBSK) -> Result<VectorLWE, CryptoAPIError> {
	let mut result: VectorLWE = VectorLWE::zero(c1.dimension, c1.nb_ciphertexts)?;
	for i in 0..(c1.nb_ciphertexts) {
		result = c1.mul_from_bootstrap_nth(c2, bsk, i, i)?;
	}
	Ok(result)
}

fn bitwise_or(img1: &mut VectorLWE, img2: &VectorLWE) -> Result<(), CryptoAPIError> {
	//(*img1).add_with_padding_inplace(img2)?;	
	let new_min = [0.];
	(*img1).add_with_new_min_inplace(img2, &new_min)?;
	Ok(())
}

fn write_image(img_buffer: ImageBuffer<Luma<u8>, Vec<u8>>, filename: String) -> Result<(), ImageError>{
	println!("Writing image ...");
	img_buffer.save(filename)?;
	Ok(())
}

fn read_image(img_name: String) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>, ImageError> {
	println!("Reading image ...");
	let img = image::open(img_name)?.grayscale();
	Ok(img.into_luma8())
}

fn encrypt_image(img: ImageBuffer<Luma<u8>, Vec<u8>>, secret_key: &LWESecretKey, encoder: &Encoder) -> Result<VectorLWE, CryptoAPIError> {
	println!("Encrypting samples ...");
	let (w,h) = img.dimensions();
	let mut messages: Vec<f64> = Vec::new();
	for i in 0..w*h {
		messages.push(img.get_pixel(i%w, i/w).0[0].into());	
	}
	let enc_image = VectorLWE::encode_encrypt(&secret_key, 
										&messages, &encoder);
	return enc_image;
}

fn decrypt_cipher_text(c: VectorLWE, sk: &LWESecretKey) -> Result<Vec<f64>, CryptoAPIError> {
	println!("Decrypting samples ...");
	let messages: Vec<f64> = c.decrypt_decode(&sk)?;	
	return Ok(messages);
}

fn prepare_image(img_data: Vec<f64>, (w,h): &(u32, u32)) -> ImageBuffer<Luma<u8>, Vec<u8>> {
	println!("Preparing image to be saved ...");
	let mut img = GrayImage::new(*w,*h);
	for (i, el) in img_data.iter().enumerate() {
		let x = (i as u32)%*w;
		let y = (i as u32)/ *w;
		img.put_pixel(x,y,Luma([*el as u8]));		
	}	
	return img;
}
