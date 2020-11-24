FROM gmlee329/isd:base

# run server
WORKDIR /ISD/
COPY . .
WORKDIR /ISD/projects/LISA/
ENTRYPOINT [ "python", "server.py" ]
EXPOSE 80