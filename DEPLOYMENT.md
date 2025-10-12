# Backend Deployment Guide (Render.com)

## Prerequisites

1. GitHub account with your repo pushed
2. Render.com account (free tier available)
3. Deepgram API key
4. OpenAI API key (optional, currently not used in real-time flow)

## Deployment Steps

### 1. Push Your Code to GitHub

```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
```

### 2. Deploy to Render.com

#### Option A: Using render.yaml (Recommended - Infrastructure as Code)

1. Go to https://render.com/dashboard
2. Click **"New"** → **"Blueprint"**
3. Connect your GitHub repository
4. Render will automatically detect `render.yaml`
5. Set environment variables:
   - `DEEPGRAM_API_KEY`: Your Deepgram API key
   - `OPENAI_API_KEY`: Your OpenAI API key (optional)
6. Click **"Apply"**

#### Option B: Manual Setup

1. Go to https://render.com/dashboard
2. Click **"New"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `interview-assistant-api`
   - **Environment**: `Python 3`
   - **Region**: `Oregon (US West)` or closest to you
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn server.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Starter` (free tier)

5. Add environment variables:
   - `DEEPGRAM_API_KEY`: Your Deepgram API key
   - `OPENAI_API_KEY`: Your OpenAI API key (optional)
   - `ALLOWED_ORIGINS`: Your frontend URL (e.g., `https://yourapp.vercel.app`)

6. Click **"Create Web Service"**

### 3. Configure Frontend to Use Production Backend

Update your frontend `.env` file:

```bash
VITE_WS_URL=wss://your-app-name.onrender.com/stream
```

**Note**: Change `ws://` to `wss://` for secure WebSocket connection in production.

### 4. Enable CORS for Your Frontend Domain

On Render dashboard, add environment variable:

```
ALLOWED_ORIGINS=https://your-frontend-domain.com,https://www.your-frontend-domain.com
```

## Monitoring & Logs

- **View Logs**: Render Dashboard → Your Service → Logs tab
- **Health Check**: `https://your-app-name.onrender.com/` (should return 404, that's normal)
- **WebSocket Test**: Use browser console to test WebSocket connection

## Troubleshooting

### Issue: WebSocket connection fails

**Solution**: Make sure you're using `wss://` (not `ws://`) in production

### Issue: CORS errors

**Solution**: Add your frontend domain to `ALLOWED_ORIGINS` environment variable

### Issue: Service keeps restarting

**Solution**: Check logs for missing environment variables or import errors

### Issue: Cold starts (free tier)

**Solution**: Render free tier spins down after 15 minutes of inactivity. First request takes ~30 seconds. Upgrade to paid plan for always-on service.

## Production Optimizations

### 1. Use Production ASGI Server

Already configured with `uvicorn[standard]` which includes:
- `httptools` for faster HTTP parsing
- `uvloop` for faster event loop

### 2. Add Health Check Endpoint

Add to `server/main.py`:

```python
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "interview-assistant-api"}
```

### 3. Add Logging Configuration

Consider adding structured logging:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Cost Considerations

### Render.com Free Tier Limits:
- 750 hours/month (enough for 1 service running 24/7)
- Spins down after 15 min of inactivity
- 512MB RAM
- Shared CPU

### Deepgram Costs:
- Nova-3 model: ~$0.0043/minute
- Typical interview (1 hour): ~$0.26
- 100 interviews/month: ~$26

### Recommended Paid Plans (Optional):
- Render Starter: $7/month (no cold starts, dedicated resources)
- Deepgram Growth: $25/month + usage

## Next Steps

1. ✅ Deploy backend to Render
2. 🔄 Deploy frontend (Vercel, Netlify, or Render Static Site)
3. 🔄 Set up custom domain (optional)
4. 🔄 Add database for storing interview transcripts
5. 🔄 Implement authentication
