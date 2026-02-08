# Pilot Data Collector - Docker Deployment

This Docker setup runs the pilot data collection web app that saves activity labels to a local volume.

## Quick Start

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up -d --build
   ```

2. **Access the app:**
   - Local: http://localhost:8000
   - Network: http://YOUR_SERVER_IP:8000

3. **Data Storage:**
   - Files are saved to Docker volume `pilot-data`
   - Mounted at `/mnt/pilot-app` inside container
   - CSV format: `Label,StartTimestamp_Unix_Ms,EndTimestamp_Unix_Ms`

## File Management

**View saved files:**
```bash
# List files in the volume
docker exec pilot-data-collector ls -la /mnt/pilot-app

# Copy files from container to host
docker cp pilot-data-collector:/mnt/pilot-app ./collected-data
```

**Access volume directly:**
```bash
# Inspect volume location
docker volume inspect pilot-data

# On Linux/WSL2, you can access directly:
# /var/lib/docker/volumes/pilot-data/_data
```

## Stopping the Service

```bash
# Stop the container
docker-compose down

# Remove volume (CAUTION: deletes all data)
docker-compose down -v
```

## Example Workflow

1. Deploy this Docker container on your server
2. Use the web interface from any device on the network
3. Collect activity labels (saved as CSV files in the volume)
4. Later, run a separate job to upload CSV files to Azure Storage
5. Clean up processed files from the volume

## File Format

Each session creates a file like:
```
activity_session_20260208_143022_a1b2c3d4.csv
```

Content:
```csv
Label,StartTimestamp_Unix_Ms,EndTimestamp_Unix_Ms
Walking,1707397822123,1707397852456
Standing,1707397852456,1707397882789
```