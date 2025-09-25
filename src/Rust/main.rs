use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_ndarray::NdArrayBackend;
use image::ImageReader;
mod generated_model;
use generated_model::Model;


fn main() -> anyhow::Result<()> {
    // --- 1) Charger l'image ---
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <image_path>", args[0]);
        return Ok(());
    }
    let img_path = &args[1];

    // Lire et redimensionner l'image
    let img = ImageReader::open(img_path)?.decode()?;
    let img = img.resize_exact(128, 128, image::imageops::FilterType::Nearest);

    // --- 2) Convertir l'image en Tensor ---
    let img_rgb = img.to_rgb8();
    let data: Vec<f32> = img_rgb
        .pixels()
        .flat_map(|p| [p[0], p[1], p[2]])
        .map(|v| v as f32 / 255.0) // normalisation [0,1]
        .collect();

    // Tensor shape: [1, 128, 128, 3] (NHWC)
    let tensor = Tensor::<NdArrayBackend<f32>, 4>::from_data(data)
        .reshape([1, 128, 128, 3]);

    // --- 3) Créer le device et charger le modèle ---
    let device = <NdArrayBackend<f32> as Backend>::Device::default();
    let model = Model::<NdArrayBackend<f32>>::from_file(
        "../generated_model/chien_vs_chat_opset16.json",
        &device,
    );

    // --- 4) Inference ---
    let output = model.forward(tensor);

    // --- 5) Récupération du vecteur de logits ---
    let logits: Vec<f32> = output.into_data()?;

    // --- 6) Décision finale ---
    let predicted_index = if logits[0] >= 0.5 { 1 } else { 0 };
    let label = match predicted_index {
        0 => "chat",
        1 => "chien",
        _ => "inconnu",
    };

    println!("Image '{}' => {}", img_path, label);

    Ok(())
}
