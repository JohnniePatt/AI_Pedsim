# Framework Design for Image-based AI Surrogate Model

กรอบแนวคิดนี้ถูกออกแบบมาเพื่ออธิบายขั้นตอนการทำงานร่วมกันของการดึงชุดข้อมูลเชิงภาพ (Image-to-Image translation) ในส่วนของ AI Surrogate Model เพื่อประเมินความหนาแน่นของผู้สัญจรในอาคาร 

แผนภูมิต่อไปนี้เขียนด้วยไวยากรณ์ Mermaid คุณสามารถนำไปใส่ในแอปพลิเคชันที่รองรับเพื่อแสดงผลเป็นแผนภาพเวิร์กโฟลว์ได้ทันที:

```mermaid
graph TD
    %% Styling
    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef generate fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef model fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef eval fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px;

    %% 1. Input Source
    subgraph Data_Source ["1. Input Physical & Simulation Data"]
        A["Physical Geometry<br>(geo_room.json, geo_door.json)"]
        B["Route Metadata<br>(start_node, end_node, route_idx)"]
        C["Simulation SQLite Data<br>(JuPedSim trajectories)"]
    end
    class A,B,C input;

    %% 2. Image Representation
    subgraph Image_Representation ["2. Image Generation & Channel Mapping"]
        A --> D["Channel R: Wall & Obstacle Boundaries"]
        B --> E["Channel G: Spawn Node Area"]
        B --> F["Channel B: Exit Node Area"]
        C --> G["Target Image: Cumulative Density Map (Grayscale)"]
        
        D & E & F --> H["Input Image A (3-Channel RGB, 256x256)"]
        G --> I["Target Image B (1-Channel Grayscale, 256x256)"]
    end
    class D,E,F,G,H,I generate;

    %% 3. AI surrogate training
    subgraph AI_Models ["3. Image-to-Image AI Surrogate Models"]
        H --> J["Plain U-Net<br>(L1 Loss, Skip Connections)"]
        H --> K["Pix2PixHD GAN<br>(Adversarial Loss, Multi-scale D)"]
        H --> L["Pix2PixHD No_D<br>(Density-Aware L1 Loss)"]
        H --> M["Conditional VAE (CVAE)<br>(Reconstruction + KL Loss)"]
    end
    class J,K,L,M model;

    %% 4. Evaluation
    subgraph Model_Evaluation ["4. Quantitative & Qualitative Evaluation"]
        J --> N["Evaluation metrics:<br>MAE, MSE, RMSE, SSIM, PSNR"]
        K --> N
        L --> N
        M --> N
        N --> O["Comparison: Run Summary & Per-Image Winners"]
    end
    class N,O eval;
```
