const express = require('express');
const { createMessage, getMessagesByConversationId, getMessageById, updateMessage, deleteMessage } = require('../../db/helperFunctions/messages');

const messagesRouter = express.Router();

// Create Message
messagesRouter.post('/', async (req, res) => {
    const { conversationId, role, content } = req.body;

    console.log("POST /messages - Request received");
    console.log("POST /messages - conversationId:", conversationId, "role:", role, "content:", content);

    if (!conversationId || !role || !content) {
        return res.status(400).json({ message: 'conversationId, role, and content are required' });
    }

    try {
        const message = await createMessage({ conversationId, role, content });
        console.log("POST /messages - Message created:", message);
        res.status(201).json(message);
    } catch (error) {
        console.error("POST /messages - Error creating message:", error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// Get All Messages by Conversation ID
messagesRouter.get('/', async (req, res) => {
    const { conversationId } = req.query;
    console.log("GET /messages - Request received for conversationId:", conversationId);

    if (!conversationId) {
        return res.status(400).json({ message: 'conversationId is required' });
    }

    try {
        const messages = await getMessagesByConversationId(conversationId);
        console.log("GET /messages - Messages fetched:", messages);
        res.status(200).json(messages);
    } catch (error) {
        console.error("GET /messages - Error fetching messages:", error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// Get Message by ID
messagesRouter.get('/:id', async (req, res) => {
    const { id } = req.params;
    console.log("GET /messages/:id - Request received for id:", id);

    try {
        const message = await getMessageById(id);
        if (message) {
            console.log("GET /messages/:id - Message fetched:", message);
            res.status(200).json(message);
        } else {
            console.log("GET /messages/:id - Message not found for id:", id);
            res.status(404).json({ message: 'Message not found' });
        }
    } catch (error) {
        console.error("GET /messages/:id - Error fetching message:", error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// Update Message
messagesRouter.put('/:id', async (req, res) => {
    const { id } = req.params;
    const fields = req.body;
    console.log("PUT /messages/:id - Request received for id:", id, "with fields:", fields);

    try {
        const updatedMessage = await updateMessage(id, fields);
        console.log("PUT /messages/:id - Message updated:", updatedMessage);
        res.status(200).json(updatedMessage);
    } catch (error) {
        console.error("PUT /messages/:id - Error updating message:", error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// Delete Message
messagesRouter.delete('/:id', async (req, res) => {
    const { id } = req.params;
    console.log("DELETE /messages/:id - Request received for id:", id);

    try {
        const deletedMessage = await deleteMessage(id);
        console.log("DELETE /messages/:id - Message deleted:", deletedMessage);
        res.status(200).json(deletedMessage);
    } catch (error) {
        console.error("DELETE /messages/:id - Error deleting message:", error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

module.exports = messagesRouter;