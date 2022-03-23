extern crate image;

use std::env;
use std::time::{Duration, Instant};
use std::path::Path;
use concrete::*;
use image::*;

const LWEKEY_FILE_NAME: &str = "my_lwe_secret_key.json";
const RLWEKEY_FILE_NAME: &str = "my_rlwe_secret_key.json";
const BSK_FILE_NAME: &str = "my_bootstrapping_key.json";

fn main() {
	let img_path : Vec<String> = env::args().collect();
	//let operation = img_path.get(2).unwrap();
	let complete_img_path = img_path.get(2).unwrap();
	let (file_name, file_ext) = complete_img_path.split_once('.').unwrap();

	// Secret keys
	let sk_rlwe = RLWESecretKey::new(&RLWE80_512_1);
	let sk = LWESecretKey::new(&LWE80_630);
	let sk_out = sk_rlwe.to_lwe_secret_key();


	//Initiating Encoder
	//let encoder_input_sobel = Encoder::new(-5., 5., 7, 0).expect("Couldn't instantiate encoder");
	//let encoder_input_average = Encoder::new(-9., 9., 7, 0).expect("Couldn't instantiate encoder");
	let encoder_input_sharp = Encoder::new(-8., 8., 11, 0).expect("Couldn't instantiate encoder");
	//let encoder_output = Encoder::new(0., 8., 4, 1).expect("Couldn't instantiate encoder");
	//Decoding image data
	let mut img_buff_1 = read_image(complete_img_path).expect("Couldn't read image");

	let result_bench;
	let mut now = Instant::now();
	//result_bench = sobel_benchmark(&img_buff_1).expect("Coulsn't calculate benchmark");
	result_bench = sharp_benchmark(&img_buff_1).expect("Coulsn't calculate benchmark");
	//result_bench = average_benchmark(&img_buff_1).expect("Coulsn't calculate benchmark");
	let benchmark_time = now.elapsed().as_millis() as f64 / 1000.;
	write_image(&img_buff_1, format!("{}_greyscale.{}", file_name, file_ext).as_str()).expect("Couldn't write benchmark image");
	write_image(&result_bench, format!("{}_Benchmark.{}", file_name, file_ext).as_str()).expect("Couldn't write benchmark image");
	let (w, h) = img_buff_1.dimensions();
	println!("Image opened succesfully\nDimensions: {}x{} px", w, h);

	// Encrpyting image data
	let mut now = Instant::now();
	let mut ciphertxt;
		//ciphertxt = encrypt_image(img_buff_1, &sk, &encoder_input_sobel).expect("Coudln' t encrpyt image");
		//ciphertxt = encrypt_image(img_buff_1, &sk, &encoder_input_average).expect("Coudln' t encrpyt image");
		ciphertxt = encrypt_image(img_buff_1, &sk, &encoder_input_sharp).expect("Coudln' t encrpyt image");
	let encryption_time = now.elapsed().as_millis() as f64 / 1000.;

	//Perform Edge Detection
	// Bootstrappin key
	/*
	println!("Creating bootstrapping key ...");
	let bsk: LWEBSK;
	let path_bsk = Path::new(BSK_FILE_NAME);
 	if path_bsk.exists() {
		println!("Bootstrapping Key found: loading {} ...", BSK_FILE_NAME);
		bsk = LWEBSK::load(BSK_FILE_NAME);
	}
	else {
		println!("No bootstrapping key file found: creating a new key ...");
		bsk = LWEBSK::new(&sk, &sk_rlwe, 3, 8);
		println!("Bootstrapping key created, saving the key in {}", BSK_FILE_NAME);
		bsk.save("my_bootstrapping_key.json");
	}*/

	now = Instant::now();
	let output_img;
		//output_img = edge_detection_no_bs(&ciphertxt, &(w as usize), &(h as usize), &sk).expect("Couldn't execute edge detection");
		//output_img = average_filter(&ciphertxt, &(w as usize), &(h as usize)).expect("Couldn't execute gaussian filter");
		output_img = sharp_filter(&ciphertxt, &(w as usize), &(h as usize)).expect("Couldn't execute gaussian filter");
	let operation_time = now.elapsed().as_millis() as f64 / 1000.;

	// Decrypting samples
	now = Instant::now();
	let data1:Vec<f64>;
	//let data1: Vec<f64> = decrypt_cipher_text(output_img, &sk_out).expect("");
		//data1 = decrypt_cipher_text_no_bs(output_img, &sk).expect("");
		//data1 = decrypt_cipher_text_average_filter(output_img, &sk).expect("Couldn't decrypt data");
		data1 = decrypt_cipher_text_sharp_filter(output_img, &sk).expect("");
	let decryption_time = now.elapsed().as_millis() as f64 / 1000.;

	// Encoding result image
	let img_output = encode_image(data1, &(w-2,h-2));

	// Write image to file
	write_image(&img_output, format!("{}_TFHE_sharp.{}", file_name, file_ext).as_str()).expect("Couldn't write image");

	let total_time = encryption_time + decryption_time + operation_time;
	println!("\n\n##############################################################################################
				BENCHMARKS TIMES:
	Encryption:	{}s
	Operation: 	{}s
	Decryption: {}s\n
	TOTAL TIME: {}s
	BENCHMARK TIME: {}s\n\n", encryption_time, operation_time, decryption_time, total_time, benchmark_time);
}

fn average_filter(img: &Vec<LWE>, w: &usize, h:&usize) -> Result<Vec<LWE>, CryptoAPIError> {
	println!("Executing Average smoothing");
	let mut output = Vec::<LWE>::new();
	let gaussian_filter = 
	[[1, 1, 1],
	[1, 1, 1],
	[1, 1, 1]];
	let mut i: usize = 0;
	let mut out_px: usize = 1;
	let out_w = w - 2*(gaussian_filter.len()/2);
	let out_h = h - 2*(gaussian_filter.len()/2);
	while out_px < out_w*out_h+1 {
		let res = convolution(img, &i, &(*w as usize), &gaussian_filter)?;
		output.push(res);
		if out_px%(out_w) == 0 {i += gaussian_filter.len()}
		else {i+= 1;}
		out_px +=1;
	}
	Ok(output)
}

fn sharp_filter(img: &Vec<LWE>, w: &usize, h:&usize) -> Result<Vec<LWE>, CryptoAPIError> {
	println!("Executing Sharpening smoothing");
	let mut output = Vec::<LWE>::new();
	let gaussian_filter = 
	[[0, -1, 0],
	[-1, 8, -1],
	[0, -1, 0]];
	let mut i: usize = 0;
	let mut out_px: usize = 1;
	let out_w = w - 2*(gaussian_filter.len()/2);
	let out_h = h - 2*(gaussian_filter.len()/2);
	while out_px < out_w*out_h+1 {
		let res = convolution(img, &i, &(*w as usize), &gaussian_filter)?;
		output.push(res);
		if out_px%(out_w) == 0 {i += gaussian_filter.len()}
		else {i+= 1;}
		out_px +=1;
	}
	Ok(output)
}

fn edge_detection_no_bs(img: &Vec<LWE>, w: &usize, h: &usize, sk: &LWESecretKey) -> Result<Vec<(LWE, LWE)>, CryptoAPIError> {
	println!("Executing Edge Detection");
	let mut output = Vec::<(LWE, LWE)>::new();
	let sobel_filter = 
	[[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]];
	let mut i: usize = 0;
	let mut out_px: usize = 1;
	let out_w = w - 2*(sobel_filter.len()/2);
	let out_h = h - 2*(sobel_filter.len()/2);
	while out_px < out_w*out_h+1 {
		let (mut Gx, mut Gy) = convolutionXY(img, &i, &(*w as usize), &sobel_filter, sk)?;
		output.push((Gx, Gy));
		if out_px%(out_w)== 0 { i += sobel_filter.len();}
		else {i+= 1;}
		out_px +=1;
	}
	Ok(output)
}

fn sharp_benchmark(img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>, ImageError> {
	println!("Executing average filter benchmark");
	let (w, h) = (*img).dimensions();
	let sobel_filter = 
	[[0,-1,0],
	[-1,8,-1],
	[0,-1,0]];
	let out_w = (w as usize) -2*(sobel_filter.len()/2);
	let out_h = (h as usize) - 2*(sobel_filter.len()/2);
	let mut result = GrayImage::new(out_w as u32, out_h as u32);
	
	let mut i:usize = 0;
	let mut out_px: usize = 1;
	while out_px < out_w*out_h+1 {
		let res = bench_convolution(img, &i, &(w as usize), &sobel_filter)?;
		let value = res/8.;
		let x = ((out_px-1)%(out_w)) as u32;
		let y = ((out_px-1)/(out_w)) as u32;
		result.put_pixel(x, y, Luma([(value*255.) as u8]));

		if out_px%(out_w)==0 {i += sobel_filter.len();}
		else {i+=1;}
		out_px +=1;
	}
	Ok(result)
}

fn average_benchmark(img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>, ImageError> {
	println!("Executing average filter benchmark");
	let (w, h) = (*img).dimensions();
	let sobel_filter = 
	[[1,1,1],
	[1,1,1],
	[1,1,1]];
	let out_w = (w as usize) -2*(sobel_filter.len()/2);
	let out_h = (h as usize) - 2*(sobel_filter.len()/2);
	let mut result = GrayImage::new(out_w as u32, out_h as u32);
	
	let mut i:usize = 0;
	let mut out_px: usize = 1;
	while out_px < out_w*out_h+1 {
		let res = bench_convolution(img, &i, &(w as usize), &sobel_filter)?;
		let value = res/9.;
		let x = ((out_px-1)%(out_w)) as u32;
		let y = ((out_px-1)/(out_w)) as u32;
		result.put_pixel(x, y, Luma([(value*255.) as u8]));

		if out_px%(out_w)==0 {i += sobel_filter.len();}
		else {i+=1;}
		out_px +=1;
	}
	Ok(result)

}

fn sobel_benchmark(img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>, ImageError> {
	println!("Executing Sobel filter benchmark");
	let (w, h) = (*img).dimensions();
	let sobel_filter = 
	[[-1,0,1],
	[-2,0,2],
	[-1,0,1]];
	let out_w = (w as usize) -2*(sobel_filter.len()/2);
	let out_h = (h as usize) - 2*(sobel_filter.len()/2);
	let mut result = GrayImage::new(out_w as u32, out_h as u32);
	
	let mut i:usize = 0;
	let mut out_px: usize = 1;
	while out_px < out_w*out_h+1 {
		let (gx, gy) = bench_convolutionXY(img, &i, &(w as usize), &sobel_filter)?;
		let (t1, t2) = (gx.round(), gy.round());
		let value = f64::sqrt(t1*t1 + t2*t2)/4.;
		let x = ((out_px-1)%(out_w)) as u32;
		let y = ((out_px-1)/(out_w)) as u32;
		result.put_pixel(x, y, Luma([(value*255.) as u8]));

		if out_px%(out_w)==0 {i += sobel_filter.len();}
		else {i+=1;}
		out_px +=1;
	}
	Ok(result)
}

fn bench_convolution(img: &ImageBuffer<Luma<u8>, Vec<u8>>, current_px: &usize, width: &usize, filter: &[[i32; 3]; 3]) -> Result<f64, ImageError>{
	let mut gx:f64 = 0.0;
	let mut gy:f64 = 0.0;
	for i in 0..filter.len() {
		for j in 0..filter[0].len() {
			let x = current_px%width + j;
			let y = current_px/width + i;
			let rotated:f64 = (img.get_pixel(x as u32, y as u32).0[0] as f64)/255.;
			let result = rotated*filter[i][j] as f64;

			if i==0 && j==0 {
				gx = result;	
			}
			else {
				gx += result;
			}
		}
	}
	Ok(gx)
}

fn bench_convolutionXY(img: &ImageBuffer<Luma<u8>, Vec<u8>>, current_px: &usize, width: &usize, filter: &[[i32; 3]; 3]) -> Result<(f64, f64), ImageError>{
	let mut gx:f64 = 0.0;
	let mut gy:f64 = 0.0;
	for i in 0..filter.len() {
		for j in 0..filter[0].len() {
			let x = current_px%width + j;
			let y = current_px/width + i;
			let rotated:f64 = (img.get_pixel(x as u32, y as u32).0[0] as f64)/255.;
			let horizontal = rotated*filter[i][j] as f64;
			let vertical = rotated*filter[j][i] as f64;

			if i==0 && j==0 {
				gx = horizontal;	
				gy = vertical;	
			}
			else {
				gx += horizontal;
				gy += vertical;
			}
		}
	}
	Ok((gx, gy))
}

/*
fn edge_detection(img: &Vec<LWE>, w: &usize, h: &usize, bsk: &LWEBSK, encoder: &Encoder, sk: &LWESecretKey) -> Result<Vec<LWE>, CryptoAPIError> {
	println!("Executing Edge Detection");
	let mut output = Vec::<LWE>::new();
	let sobel_filter = 
	[[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]];
	let mut i: usize = 0;
	let mut out_px: usize = 1;
	while out_px < (w-2)*(h-2)+1 {
		let now = Instant::now();
		let (mut Gx, mut Gy) = convolutionXY(img, &i, &(*w as usize), &sobel_filter, sk)?;
		Gx = Gx.bootstrap_with_function(bsk, |x| f64::abs(x.round()), encoder)?;
		Gy = Gy.bootstrap_with_function(bsk, |x| f64::abs(x.round()), encoder)?;
		let Gtot = Gx.add_with_padding_exact(&Gy)?;
		println!("|Gx| |Gy|-> {} {}", Gx.decrypt_decode(sk_out)?, Gy.decrypt_decode(sk_out)?);
		println!("|Gtot|-> {}", Gtot.decrypt_decode(sk_out)?);
		println!("Pixel {}: duration: {} secs", i as i32, now.elapsed().as_millis() as f64 / 1000.);
		output.push(Gx);
		if out_px%(*w-2)== 0 { i += 3;}
		else {i+= 1;}
		out_px +=1;
	}
	Ok(output)
}*/

fn convolution(img: &Vec<LWE>, current_px: &usize, width: &usize, filter: &[[i32; 3]; 3]) -> Result<LWE, CryptoAPIError>{
	let mut tot = LWE::zero(256)?;
	for i in 0..filter.len() {
		for j in 0..filter[0].len() {
			let rotated = &img[current_px + i*width + j];
			let coefficient = filter[j][i];
			let temp = rotated.mul_constant_static_encoder(coefficient)?;
			//println!("temp: {}", temp.decrypt_decode(sk)?);
			if i==0 && j==0 {
				tot = temp;
			}
			else {
				tot.add_centered_inplace(&temp)?;
			}
			//println!("tot: {}", tot.decrypt_decode(sk)?);
		}
	}
	//println!("-------- tot: {}", tot.decrypt_decode(sk)?);
	Ok(tot)
}
fn convolutionXY(img: &Vec<LWE>, current_px: &usize, width: &usize, filter: &[[i32; 3]; 3], sk: &LWESecretKey) -> Result<(LWE, LWE), CryptoAPIError>{
	let mut Ix = LWE::zero(256)?;
	let mut Iy = LWE::zero(256)?;
	for i in 0..filter.len() {
		for j in 0..filter[0].len() {
			let rotated = &img[current_px + i*width + j];
			//println!("px: {}", current_px + i*width + j);
			let horizontal = rotated.mul_constant_static_encoder(filter[i][j])?;
			let vertical = rotated.mul_constant_static_encoder(filter[j][i])?;
			//println!("px py: {} {}", horizontal.decrypt_decode(sk)?, vertical.decrypt_decode(sk)?);
			if i==0 && j==0 {
				Ix = horizontal;
				Iy = vertical;
			}
			else {
				Ix.add_centered_inplace(&horizontal)?;
				Iy.add_centered_inplace(&vertical)?;
			}
			//println!("Ix Iy: {} {}", Ix.decrypt_decode(sk)?, Iy.decrypt_decode(sk)?);
		}
	}
	//println!("(Ix, Iy): ({}, {})", Ix.decrypt_decode(sk)?, Iy.decrypt_decode(sk)?);
	Ok((Ix, Iy))
}


fn write_image(img_buffer: &ImageBuffer<Luma<u8>, Vec<u8>>, filename: &str) -> Result<(), ImageError>{
	println!("Writing image at {}", filename);
	img_buffer.save(filename)?;
	Ok(())
}

fn read_image(img_name: &str) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>, ImageError> {
	println!("Decoding image at: {} ...", img_name);
	let img = image::open(img_name)?.grayscale();
	Ok(img.into_luma8())
}

fn encrypt_image(img: ImageBuffer<Luma<u8>, Vec<u8>>, secret_key: &LWESecretKey, encoder: &Encoder) -> Result<Vec<LWE>, CryptoAPIError> {
	println!("Encrypting image data ...");
	let (w,h) = img.dimensions();
	let mut enc_image: Vec<LWE> = Vec::new();
	for i in 0..w*h {
		let value : f64= img.get_pixel(i%w, i/w).0[0].into();
		//if (i as u32)%w < 3 && (i as u32)/w < 3 {println!("{}", value);}
		let sample = LWE::encode_encrypt(secret_key, value/255., encoder)?;
		//println!("value : {}", sample.decrypt_decode_round(secret_key)?);
		enc_image.push(sample);
	}
	Ok(enc_image)
}

fn decrypt_cipher_text_sharp_filter(c: Vec<LWE>, sk: &LWESecretKey) -> Result<Vec<f64>, CryptoAPIError> {
	println!("Decrypting samples ...");
	let mut messages: Vec<f64> = Vec::new();
	for i in 0..c.len() {
		let t1 = c[i].decrypt_decode(sk)?;
		let value = t1/8.;
		//println!("value: {:.2}", value);
		messages.push(value);
	}
	return Ok(messages);
}

fn decrypt_cipher_text_average_filter(c: Vec<LWE>, sk: &LWESecretKey) -> Result<Vec<f64>, CryptoAPIError> {
	println!("Decrypting samples ...");
	let mut messages: Vec<f64> = Vec::new();
	for i in 0..c.len() {
		let t1 = c[i].decrypt_decode(sk)?;
		let value = t1/9.;
		//println!("value: {:.2}", value);
		messages.push(value);
	}
	return Ok(messages);
}

fn decrypt_cipher_text_no_bs(c: Vec<(LWE, LWE)>, sk: &LWESecretKey) -> Result<Vec<f64>, CryptoAPIError> {
	println!("Decrypting samples ...");
	let mut messages: Vec<f64> = Vec::new();
	for i in 0..c.len() {
		let (t1, t2) = (c[i].0.decrypt_decode(sk)?, c[i].1.decrypt_decode(sk)?);
		//let threashold = 0.274509804; // 27% circa
		let threashold = 0.302941176; // 35% circa
		let (t1, t2) = (t1.round(), t2.round());
		let value = f64::sqrt(t1*t1 + t2*t2)/4.;
		//messages.push(if value < 0.4 {0.} else {value});
		//messages.push(f64::max(value, threashold));
		messages.push(value);
	}
	//println!("{:?}", messages);
	return Ok(messages);
}

fn decrypt_cipher_text(c: Vec<LWE>, sk: &LWESecretKey) -> Result<Vec<f64>, CryptoAPIError> {
	println!("Decrypting samples ...");
	let mut messages: Vec<f64> = Vec::new();
	for i in 0..c.len() {
		messages.push(c[i].decrypt_decode_round(sk)?);
	}
	println!("{:?}", messages);
	return Ok(messages);
}

fn encode_image(img_data: Vec<f64>, (w,h): &(u32, u32)) -> ImageBuffer<Luma<u8>, Vec<u8>> {
	println!("Preparing image to be saved ...");
	let mut img = GrayImage::new(*w,*h);
	for (i, el) in img_data.iter().enumerate() {
		let x = (i as u32)%*w;
		let y = (i as u32)/ *w;
		let value = (*el*255.) as u8;
		//println!("{}", value);
		//let value = (*el*255.) as u8;
		//if (i as u32)%w < 3 && (i as u32)/w < 3 {println!("{}", value);}
		img.put_pixel(x,y,Luma([value]));	
	}	
	println!("Image encoded succesfully");
	return img;
}
