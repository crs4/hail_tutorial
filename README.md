# Hail Tutorial  

This tutorial provides Jupyter notebooks based on the [**Hail GWAS tutorial**](https://hail.is/docs/0.2/tutorials/01-genome-wide-association-study.html) to demonstrate how to perform a GWAS (Genome-Wide Association Study) analysis using a VCF file, while storing Hail data structures in an external S3 storage.  

---

## 📌 Setting Up the Environment  

To get started, clone this repository:  

```bash
git clone https://github.com/crs4/hail_tutorial.git
cd hail_tutorial
```

Then, start the Docker environment:  

```bash
docker compose up -d
```

On the first run, Docker will build two images:  

- **`hail_tutorial`** – The environment for running the tutorials in a Jupyter Lab server.  
- **`minio`** – A high-performance object storage service with an API compatible with Amazon S3.  

To shut down the Docker containers, run:  

```bash
docker compose down
```

---

## 🚀 Accessing the Tutorials  

### Jupyter Lab  

- Open a browser and go to **[localhost:18888](http://localhost:18888)**.  
- Enter the password: **`12345678`** (only required the first time).  

### MinIO S3 Storage  

- Open **[localhost:9001](http://localhost:9001)** in a browser.  
- Use the credentials:  
  - **Username**: `root`  
  - **Password**: `passpass`  

Once you run a Jupyter notebook, the `data-hail` bucket will be created in MinIO.  

---

## 📂 Jupyter Notebooks  

The `notebooks` folder contains two Jupyter notebooks:

- [**Hail_tutorial-GWAS-vcf.ipynb**](notebooks/Hail_tutorial-GWAS-vcf.ipynb):  
  A complete GWAS analysis using a VCF file.  
- [**Hail_tutorial-GWAS-vcf-smaller.ipynb**](notebooks/Hail_tutorial-GWAS-vcf-smaller.ipynb):  
  A smaller version of the previous notebook.  

---

## ⚠️ Additional Notes  

- Ensure Docker compose is installed and running before starting the environment.  
- If you encounter issues accessing Jupyter Lab, check if the container is running:  

  ```bash
  docker compose ps
  ```  

- Restart the Docker environment if needed:  

  ```bash
  docker compose down && docker compose up -d
  ```  

- Print logs of the Jupyter Hub service if needed:

  ```bash
  docker compose logs -f jupyter
  ```
