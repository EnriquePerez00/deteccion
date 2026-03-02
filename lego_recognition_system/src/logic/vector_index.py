import numpy as np
import pickle
import os
import faiss

class VectorIndex:
    def __init__(self, index_path=None, dim=384): # DINOv2 ViT-Small is 384
        self.dim = dim
        self.metadata = [] # List of {ldraw_id: ...}
        
        # Initialize FAISS Index (Inner Product for Cosine Similarity if L2 normalized)
        self.index = faiss.IndexFlatIP(self.dim)
        
        # Try to move to GPU if T4 is available
        self.res = None
        try:
            self.res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
            self.use_gpu = True
            print("🚀 FAISS Index initialized on GPU.")
        except Exception as e:
            self.use_gpu = False
            print("⚠️ FAISS GPU not available, falling back to CPU.")
            
        if index_path and os.path.exists(index_path):
            self.load(index_path)

    @property
    def embeddings(self):
        # Compatibility property for the UI which checks `if v_index.embeddings:`
        return [True] if self.index.ntotal > 0 else []

    def _normalize(self, x):
        """L2 normalize for Cosine Similarity equivalence in Inner Product"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-10, norms)
        return x / norms

    def add(self, embedding, metadata):
        # Normalize and cast to float32 for FAISS
        emb = np.array(embedding, dtype=np.float32)
        emb_norm = self._normalize(emb)
        self.index.add(emb_norm)
        self.metadata.append(metadata)

    def search(self, query_embedding, k=1, deduplicate=False):
        if self.index.ntotal == 0:
            return []
            
        # If deduplicate is True, we need to request more raw matches from FAISS 
        # to ensure we end up with k distinct pieces
        raw_k = k * 5 if deduplicate else k
        
        # Normalize incoming query for Cosine Simulation
        query = np.array(query_embedding, dtype=np.float32)
        query_norm = self._normalize(query)
        
        # Search returns distances (similarities for IP) and indices
        similarities, indices = self.index.search(query_norm, raw_k)
        
        results = []
        seen_ids = set()
        
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1 and idx < len(self.metadata):
                m = self.metadata[idx]
                ld_id = m.get('ldraw_id', 'Unknown')
                
                if deduplicate:
                    if ld_id in seen_ids:
                        continue
                    seen_ids.add(ld_id)
                
                results.append({
                    'metadata': m,
                    'distance': float(1.0 - similarities[0][i]),
                    'similarity': float(similarities[0][i])
                })
                
                if len(results) >= k:
                    break
                    
        return results

    def save(self, path):
        # Save Metadata and FAISS Index separately but bundle in same folder
        base, _ = os.path.splitext(path)
        meta_path = f"{base}_meta.pkl"
        idx_path = f"{base}.index"
        
        with open(meta_path, 'wb') as f:
            pickle.dump({'metadata': self.metadata}, f)
            
        # FAISS index must be on CPU to be written to disk
        cpu_index = faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index
        faiss.write_index(cpu_index, idx_path)
            
        print(f"✅ FAISS index saved with {self.index.ntotal} entries to {idx_path}")

    def load(self, path):
        base, _ = os.path.splitext(path)
        meta_path = f"{base}_meta.pkl"
        idx_path = f"{base}.index"
        
        if os.path.exists(idx_path) and os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata.extend(data['metadata'])
                
            cpu_index = faiss.read_index(idx_path)
            
            # Since FAISS doesn't allow direct merging easily without IDs, 
            # if we are merging multiple shards (like build_reference_index does), 
            # we must extract vectors and add them to our master live index
            
            if cpu_index.ntotal > 0:
                # Reconstruct vectors to add to live GPU index
                vectors = np.zeros((cpu_index.ntotal, cpu_index.d), dtype=np.float32)
                for i in range(cpu_index.ntotal):
                    vectors[i] = cpu_index.reconstruct(i)
                self.index.add(vectors)
                
            print(f"📂 Loaded {cpu_index.ntotal} vectors from {os.path.basename(idx_path)}. Total database: {self.index.ntotal}")
            return True
        else:
            # Fallback for old .pkl compatibility
            if os.path.exists(path) and path.endswith('.pkl'):
                try:
                    with open(path, 'rb') as f:
                        data = pickle.load(f)
                        if 'embeddings' in data and data['embeddings']:
                            embs = np.array(data['embeddings'], dtype=np.float32)
                            self.index.add(self._normalize(embs))
                            self.metadata.extend(data['metadata'])
                            print(f"📂 Migrated {len(data['embeddings'])} vectors from Legacy Pickle {os.path.basename(path)}.")
                            return True
                except Exception as e:
                    print(f"Failed to load old legacy index: {e}")
            return False

    def get_indexed_ids(self):
        """Returns a set of all unique ldraw_ids currently in the index."""
        indexed_ids = set()
        for m in self.metadata:
            if isinstance(m, dict) and 'ldraw_id' in m:
                indexed_ids.add(m['ldraw_id'])
        return indexed_ids
