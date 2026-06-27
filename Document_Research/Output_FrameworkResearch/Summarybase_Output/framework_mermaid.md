# Framework Design for AI Surrogate Model

กรอบแนวคิดนี้ถูกออกแบบมาให้เห็นภาพรวมของงานวิจัย ซึ่งสอดคล้องกับ Code Development และข้อเสนอที่เขียนใน Document คุณสามารถนำโค้ดด้านล่างไปใส่ในเครื่องมือที่รองรับ Mermaid (เช่น Notion, GitHub, หรือ Mermaid Live Editor) เพื่อแสดงแผนภาพ (Flowchart) ได้ทันที

```mermaid
graph LR
    %% Styling
    classDef config fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef process fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef geom fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef output fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px;

    %% Configuration
    subgraph Configuration ["1. Input Configurations"]
        A[config_housegan.json] --> B[Complexity<br>Number of Rooms]
        A --> C[Room Area Mode<br>Default / Big]
        A --> D[Num Scenarios & Seed]
    end
    class A,B,C,D config;

    %% Procedural Generation
    subgraph Generation ["2. Procedural Layout Generation"]
        B --> E[Place Corridors]
        C --> E
        D --> E
        E --> F["Attach Rooms to Corridors/Rooms"]
        F --> G["Calculate Room Boundaries<br>(Shapely Polygons)"]
        G --> H[Detect Physical Doors<br>Intersection & Overlap]
    end
    class E,F,G,H process;

    %% Graph Extraction
    subgraph Graph_Construction ["3. Topological Graph Construction"]
        H --> I[Node Extraction<br>Room Area, Type]
        H --> J[Edge Extraction<br>Connected Rooms]
    end
    class I,J geom;

    %% Output
    subgraph Output ["4. Output Data Structure"]
        I --> K[(Geo_scenario/<br>Plan Directory)]
        J --> K
        K --> L[topological_graph.json]
        K --> M[geo_*.json<br>Room, Corridor, Door]
        K --> N[metadata.json & Previews]
    end
    class K,L,M,N output;
```
