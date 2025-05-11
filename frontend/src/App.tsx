import React, { useState, useCallback } from 'react';
import { Box, Container, Paper, Typography, Button, Grid, List, ListItem, ListItemText, ListItemIcon, Radio, IconButton } from '@mui/material';
import { styled } from '@mui/material/styles';
import { useDropzone } from 'react-dropzone';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import BlurOnIcon from '@mui/icons-material/BlurOn';
import DeleteIcon from '@mui/icons-material/Delete';
import CropIcon from '@mui/icons-material/Crop';
import FaceIcon from '@mui/icons-material/Face';
import DirectionsCarIcon from '@mui/icons-material/DirectionsCar';
import DownloadIcon from '@mui/icons-material/Download';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

const UploadBox = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  textAlign: 'center',
  cursor: 'pointer',
  border: `2px dashed ${theme.palette.primary.main}`,
  backgroundColor: theme.palette.background.default,
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
  },
}));

const Canvas = styled(Box)(({ theme }) => ({
  width: '100%',
  height: '500px',
  backgroundColor: theme.palette.grey[100],
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  position: 'relative',
  overflow: 'hidden',
}));

const PreviewBox = styled(Box)(({ theme }) => ({
  position: 'absolute',
  border: `2px solid ${theme.palette.primary.main}`,
  backgroundColor: 'rgba(25, 118, 210, 0.1)',
}));

interface Detection {
  class: string;
  confidence: number;
  bbox: number[];
  size: {
    width: number;
    height: number;
  };
}

function App() {
  const [image, setImage] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [selectedObject, setSelectedObject] = useState<string | null>(null);
  const [hoveredObject, setHoveredObject] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/upload`, formData);
      setImage(URL.createObjectURL(file));
      setDetections(response.data.detections);
    } catch (error) {
      console.error('Error uploading image:', error);
    }
  }, []);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    multiple: false
  });

  const handleObjectSelect = (objectClass: string) => {
    setSelectedObject(objectClass === selectedObject ? null : objectClass);
  };

  const handleObjectHover = (objectClass: string | null) => {
    setHoveredObject(objectClass);
  };

  const applyEffect = async (effectType: string, isGlobal: boolean = false) => {
    if (!image) return;

    try {
      const response = await axios.post(`${API_URL}/apply-effect`, {
        filename: image.split('/').pop(),
        effect_type: effectType,
        target_objects: isGlobal ? null : [selectedObject],
        global_effect: isGlobal
      });

      setProcessedImage(`${API_URL}/download/${response.data.processed_filename}`);
    } catch (error) {
      console.error('Error applying effect:', error);
    }
  };

  const downloadImage = async (highRes: boolean = false) => {
    if (!processedImage) return;

    try {
      const response = await axios.get(`${API_URL}/download/${processedImage.split('/').pop()}`, {
        params: { high_res: highRes },
        responseType: 'blob'
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `blur-ai-${Date.now()}.jpg`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Error downloading image:', error);
    }
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom align="center">
        BlurAI
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          {!image ? (
            <UploadBox {...getRootProps()}>
              <input {...getInputProps()} />
              <CloudUploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
              <Typography variant="h6">
                Drag and drop an image here, or click to select
              </Typography>
            </UploadBox>
          ) : (
            <Canvas>
              <img
                src={processedImage || image}
                alt="Uploaded"
                style={{ maxWidth: '100%', maxHeight: '100%' }}
              />
              {hoveredObject && detections.find(d => d.class === hoveredObject) && (
                <PreviewBox
                  style={{
                    left: detections.find(d => d.class === hoveredObject)?.bbox[0],
                    top: detections.find(d => d.class === hoveredObject)?.bbox[1],
                    width: detections.find(d => d.class === hoveredObject)?.size.width,
                    height: detections.find(d => d.class === hoveredObject)?.size.height,
                  }}
                />
              )}
            </Canvas>
          )}
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Detected Objects
            </Typography>
            <List>
              {detections.map((detection, index) => (
                <ListItem
                  key={index}
                  onMouseEnter={() => handleObjectHover(detection.class)}
                  onMouseLeave={() => handleObjectHover(null)}
                >
                  <ListItemIcon>
                    <Radio
                      checked={selectedObject === detection.class}
                      onChange={() => handleObjectSelect(detection.class)}
                    />
                  </ListItemIcon>
                  <ListItemText
                    primary={detection.class}
                    secondary={`Confidence: ${(detection.confidence * 100).toFixed(1)}%`}
                  />
                </ListItem>
              ))}
            </List>

            <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
              Tools
            </Typography>

            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Global Tools
              </Typography>
              <Button
                variant="contained"
                startIcon={<FaceIcon />}
                onClick={() => applyEffect('blur_faces', true)}
                sx={{ mr: 1, mb: 1 }}
              >
                Blur All Faces
              </Button>
              <Button
                variant="contained"
                startIcon={<DirectionsCarIcon />}
                onClick={() => applyEffect('blur_plates', true)}
                sx={{ mr: 1, mb: 1 }}
              >
                Blur All Plates
              </Button>
            </Box>

            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Object Tools
              </Typography>
              <Button
                variant="contained"
                startIcon={<BlurOnIcon />}
                onClick={() => applyEffect('blur')}
                disabled={!selectedObject}
                sx={{ mr: 1, mb: 1 }}
              >
                Blur
              </Button>
              <Button
                variant="contained"
                startIcon={<DeleteIcon />}
                onClick={() => applyEffect('delete')}
                disabled={!selectedObject}
                sx={{ mr: 1, mb: 1 }}
              >
                Delete
              </Button>
              <Button
                variant="contained"
                startIcon={<CropIcon />}
                onClick={() => applyEffect('crop')}
                disabled={!selectedObject}
                sx={{ mr: 1, mb: 1 }}
              >
                Crop
              </Button>
            </Box>

            {processedImage && (
              <Box sx={{ mt: 4 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Download
                </Typography>
                <Button
                  variant="contained"
                  startIcon={<DownloadIcon />}
                  onClick={() => downloadImage(false)}
                  sx={{ mr: 1 }}
                >
                  Standard
                </Button>
                <Button
                  variant="contained"
                  startIcon={<DownloadIcon />}
                  onClick={() => downloadImage(true)}
                >
                  High Resolution
                </Button>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
}

export default App; 