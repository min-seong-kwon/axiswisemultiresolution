import numpy as np

class TriMesh:
    def __init__(self):
        self.n_vertices = 0
        self.n_faces = 0
        self.vertices = np.zeros((20000, 6), dtype=np.float32)
        self.faces = np.zeros((10000, 3), dtype=np.int32)
        self.face_normals = None
        
    def setMeshData(self, vertices, faces, uvs):
        self.vertices = vertices
        self.faces = faces
        self.uvs = uvs
        
        self.n_vertices = vertices.shape[0]
        self.n_faces = faces.shape[0]
        
    def loadVertexFace(self, ply_path):
        ply_data = PlyData.read(ply_path)
        
        self.n_vertices = ply_data['vertex'].count
        self.n_faces = ply_data['face'].count
        
        self.vertices = np.zeros((self.n_vertices, 3), dtype=np.float32)
        self.faces = np.zeros((self.n_faces, 3), dtype=np.int32)
        
        for i in range(self.n_vertices):
            self.vertices[i, 0] = ply_data['vertex'].data[i][0]
            self.vertices[i, 1] = ply_data['vertex'].data[i][1]
            self.vertices[i, 2] = ply_data['vertex'].data[i][2]
            
        for i in range(self.n_faces):
            self.faces[i, 0] = ply_data['face'].data['vertex_indices'][i][0]
            self.faces[i, 1] = ply_data['face'].data['vertex_indices'][i][1]
            self.faces[i, 2] = ply_data['face'].data['vertex_indices'][i][2]
            
    def loadVertexUVFace(self, ply_path):
        ply_data = PlyData.read(ply_path)
        
        self.n_vertices = ply_data['vertex'].count
        self.n_faces = ply_data['face'].count
        
        self.vertices = np.zeros((self.n_vertices, 3), dtype=np.float32)
        self.uvs = np.zeros((self.n_vertices, 2), dtype=np.float32)
        self.faces = np.zeros((self.n_faces, 3), dtype=np.int32)
        
        for i in range(self.n_vertices):
            self.vertices[i, 0] = ply_data['vertex'].data[i][0]
            self.vertices[i, 1] = ply_data['vertex'].data[i][1]
            self.vertices[i, 2] = ply_data['vertex'].data[i][2]
            
            self.uvs[i, 0] = ply_data['vertex'].data[i][3]
            self.uvs[i, 1] = ply_data['vertex'].data[i][4]
            
        for i in range(self.n_faces):
            self.faces[i, 0] = ply_data['face'].data['vertex_indices'][i][0]
            self.faces[i, 1] = ply_data['face'].data['vertex_indices'][i][1]
            self.faces[i, 2] = ply_data['face'].data['vertex_indices'][i][2]
        
            
    def saveVertexUVFace(self, ply_path):
        out_vertices = np.hstack((self.vertices[0:self.n_vertices, 0:3], self.uvs[0:self.n_vertices, :]))
        out_faces = self.faces[0:self.n_faces, 0:3]
        
        datastr1 = ["%010f %010f %010f %010f %010f\n" % tuple(vertex) for vertex in out_vertices]
        datastr2 = ["3 %d %d %d\n" % tuple(triangle) for triangle in out_faces]
        
        with open(ply_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex %d\n" % len(out_vertices))
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float s\n")
            f.write("property float t\n")
            f.write("element face %d\n" % len(out_faces))
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            f.writelines(datastr1)
            f.writelines(datastr2)
               
    def add_vertex(self, X, Y, Z, R, G, B):
        if self.n_vertices == self.vertices.shape[0]:
            temp = np.zeros((20000, 6), np.float32)
            self.vertices = np.vstack((self.vertices, temp))
            
        self.vertices[self.n_vertices, :] = (X, Y, Z, R, G, B)
        self.n_vertices += 1
            
        return self.n_vertices - 1 # vertex 인덱스를 반환
    
    def add_vertices(self, arr):
        much = len(arr)
        while self.n_vertices + much >= self.vertices.shape[0]:
            temp = np.zeros((20000, 6), np.float32)
            self.vertices = np.vstack((self.vertices, temp))
        self.vertices[self.n_vertices:self.n_vertices+much, :] = arr
        self.n_vertices += much
        
        return self.n_vertices - 1
    
    def add_face(self, v1, v2, v3):
        if self.n_faces == self.faces.shape[0]:
            temp = np.zeros((10000, 3), np.int32)
            self.faces = np.vstack((self.faces, temp))
            
        self.faces[self.n_faces, :] = (v1, v2, v3)
        self.n_faces += 1

        return self.n_faces -1 # face 인덱스를 반환     
    
    def add_faces(self, arr):
        much = len(arr)
        while self.n_faces + much >= self.faces.shape[0]:
            temp = np.zeros((10000, 3), np.int32)
            self.faces = np.vstack((self.faces, temp))
        self.faces[self.n_faces:self.n_faces+much, :] = arr
        self.n_faces += much

        return self.n_faces -1 # face 인덱스를 반환  
    
    def add_faces_from_list(self, faces):
        for v1, v2, v3 in faces:
            self.add_face(v1, v2, v3)
    
    def calculate_face_normals(self):
        self.face_normals = np.zeros((self.n_faces, 3), np.float32)
        for f in range(self.n_faces):
            v0, v1, v2 = self.vertices[self.faces[f,:], 0:3]
            n = np.cross((v1 - v0), (v2 - v0))
            l = np.linalg.norm(n)
            if l > 0.0:
                n = n / l
            
            self.face_normals[f, :] = n
            
        if self.n_faces <= 1:
            return 0
        else:
            return np.linalg.det(np.cov(self.face_normals.T))
        
    def calculate_avg_sharpnesses(self):
        self.face_normals = np.zeros((self.n_faces, 3), np.float32)
        for f in range(self.n_faces):
            v0, v1, v2 = self.vertices[self.faces[f,:], 0:3]
            n = np.cross((v1 - v0), (v2 - v0))
            l = np.linalg.norm(n)
            if l > 0.0:
                n = n / l
            
            self.face_normals[f, :] = n
            
        normals_on_edge = {}
        
        for f in range(self.n_faces):
            i0, i1, i2 = self.faces[f, :]
            normals_on_edge[np.minimum(i0, i1), np.maximum(i0, i1)] = []
            normals_on_edge[np.minimum(i1, i2), np.maximum(i1, i2)] = []
            normals_on_edge[np.minimum(i2, i0), np.maximum(i2, i0)] = []
            
        for f in range(self.n_faces):
            i0, i1, i2 = self.faces[f, :]
            normals_on_edge[np.minimum(i0, i1), np.maximum(i0, i1)].append(self.face_normals[f, :])
            normals_on_edge[np.minimum(i1, i2), np.maximum(i1, i2)].append(self.face_normals[f, :])
            normals_on_edge[np.minimum(i2, i0), np.maximum(i2, i0)].append(self.face_normals[f, :])
            
        w_sum = 0.0
        w_cnt = 0
        for key in normals_on_edge.keys():
            if len(normals_on_edge[key]) != 2:
                continue
            nn = np.dot(normals_on_edge[key][0], normals_on_edge[key][1])
            nn = np.minimum(1.0, np.maximum(-1.0, nn))
            
            w_sum += np.arccos(nn)
            w_cnt += 1
            
        if w_cnt <= 0:
            return 0
        else:
            sharpeness = w_sum / w_cnt
            return sharpeness
            
    def calculate_max_sharpnesses(self):
        self.face_normals = np.zeros((self.n_faces, 3), np.float32)
        for f in range(self.n_faces):
            v0, v1, v2 = self.vertices[self.faces[f,:], 0:3]
            n = np.cross((v1 - v0), (v2 - v0))
            l = np.linalg.norm(n)
            if l > 0.0:
                n = n / l
            
            self.face_normals[f, :] = n
            
        normals_on_edge = {}
        
        for f in range(self.n_faces):
            i0, i1, i2 = self.faces[f, :]
            normals_on_edge[np.minimum(i0, i1), np.maximum(i0, i1)] = []
            normals_on_edge[np.minimum(i1, i2), np.maximum(i1, i2)] = []
            normals_on_edge[np.minimum(i2, i0), np.maximum(i2, i0)] = []
            
        for f in range(self.n_faces):
            i0, i1, i2 = self.faces[f, :]
            normals_on_edge[np.minimum(i0, i1), np.maximum(i0, i1)].append(self.face_normals[f, :])
            normals_on_edge[np.minimum(i1, i2), np.maximum(i1, i2)].append(self.face_normals[f, :])
            normals_on_edge[np.minimum(i2, i0), np.maximum(i2, i0)].append(self.face_normals[f, :])
            
        sharpeness = []
        w_cnt = 0
        for key in normals_on_edge.keys():
            if len(normals_on_edge[key]) != 2:
                continue
            nn = np.dot(normals_on_edge[key][0], normals_on_edge[key][1])
            nn = np.minimum(1.0, np.maximum(-1.0, nn))
            
            sharpeness.append(np.arccos(nn))
            w_cnt += 1
            
        if w_cnt <= 0:
            return 0
        else:
            sharpeness = np.array(sharpeness)
            return sharpeness.max()

    def save_ply(self, filename):
        out_vertices = self.vertices[0:self.n_vertices, 0:6]
        out_faces = self.faces[0:self.n_faces, 0:3]

        datastr1 = ["%010f %010f %010f %d %d %d\n" % tuple(vertex) for vertex in out_vertices]
        datastr2 = ["3 %d %d %d\n" % tuple(triangle) for triangle in out_faces]
        
        with open(filename, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex %d\n" % len(out_vertices))
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("element face %d\n" % len(out_faces))
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            f.writelines(datastr1)
            f.writelines(datastr2)
            
    def save_ply_without_properties(self, filename):
        out_vertices = self.vertices[0:self.n_vertices, 0:3]
        out_faces = self.faces[0:self.n_faces, 0:3]

        datastr1 = ["%010f %010f %010f\n" % tuple(vertex) for vertex in out_vertices]
        datastr2 = ["3 %d %d %d\n" % tuple(triangle) for triangle in out_faces]
        
        with open(filename, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex %d\n" % len(out_vertices))
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("element face %d\n" % len(out_faces))
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            f.writelines(datastr1)
            f.writelines(datastr2) 
