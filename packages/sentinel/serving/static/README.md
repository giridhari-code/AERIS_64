# No UI assets in the backend

Frontend HTML lives only in **`apps/frontend/public/`**.

This directory is intentionally empty of HTML so frontend and backend stay unmixed.
With `SERVE_UI=1`, Python may serve `apps/frontend/public` for local debug only.
