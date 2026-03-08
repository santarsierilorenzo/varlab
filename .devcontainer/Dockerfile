FROM python:3.11

# Create non-root user with bash
RUN useradd -ms /bin/bash myuser

# Use a neutral workdir (VS Code will override this later)
WORKDIR /tmp

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER myuser

CMD ["bash"]
